from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from athena.amc import AMC, AMCConfig, Event, LlamaRole, ModelId
from athena.amc.config import DEFAULT_KV_CACHE_GB_TRANSLATE
from athena.amc.backends import (
    MockLlamaBackend,
    MockTranslatorBackend,
    MockWhisperBackend,
    _require_ir_dir,
)
from athena.amc.exceptions import ModelPathError, RoleError, TTSNotWiredError
from athena.amc.gpu import is_gpu_dead_error, translator_unsafe_reason
from athena.amc.roles import evaluator_system_prompt, tutor_system_prompt
from athena.amc.translate import (
    chunk_for_translator,
    format_translate_prompt,
    pack_text,
    prompt_token_limit,
    split_sentences,
)


class _SlowLlama(MockLlamaBackend):
    def __init__(self, hold: float) -> None:
        super().__init__()
        self.hold = hold
        self.started = threading.Event()

    def generate(self, user_text: str, **kwargs) -> str:  # type: ignore[no-untyped-def]
        self.started.set()
        time.sleep(self.hold)
        return super().generate(user_text, **kwargs)


class AMCTests(unittest.TestCase):
    def setUp(self) -> None:
        self.amc = AMC.mock()

    def tearDown(self) -> None:
        self.amc.shutdown()

    def test_enter_event_t_loads_only_llama(self) -> None:
        self.amc.enter_event_t({"id": "ch1-s1", "title": "Fractions"})
        st = self.amc.status()
        self.assertEqual(st.event, Event.T)
        self.assertEqual(st.role, LlamaRole.TUTOR)
        self.assertEqual(st.resident_model, ModelId.LLAMA_INT4)
        self.assertTrue(self.amc._llama.is_loaded())
        self.assertFalse(self.amc._whisper.is_loaded())

    def test_tutor_keeps_conversation(self) -> None:
        self.amc.enter_event_t({"id": "s1"})
        self.amc.tutor_ask("what is a fraction?")
        self.amc.tutor_ask("give an example")
        self.assertEqual(len(self.amc._memory), 4)
        rendered = self.amc._memory.render()
        self.assertIn("what is a fraction?", rendered)
        self.assertIn("give an example", rendered)

    def test_speak_unloads_llama_and_loads_whisper(self) -> None:
        self.amc.enter_event_t({"id": "s1"})
        self.amc.tutor_ask("explain")
        text = self.amc.transcribe([0.0] * 1600)
        self.assertEqual(text, "what does this paragraph mean")
        self.assertEqual(self.amc.status().resident_model, ModelId.WHISPER)
        self.assertFalse(self.amc._llama.is_loaded())
        self.assertTrue(self.amc._whisper.is_loaded())
        # text memory survives the GPU switch
        self.assertGreater(len(self.amc._memory), 0)

    def test_tutor_after_whisper_reloads_llama_with_history(self) -> None:
        self.amc.enter_event_t({"id": "s1"})
        self.amc.tutor_ask("first doubt")
        self.amc.transcribe([0.0] * 100)
        self.amc.tutor_ask("second doubt")
        st = self.amc.status()
        self.assertEqual(st.resident_model, ModelId.LLAMA_INT4)
        self.assertFalse(self.amc._whisper.is_loaded())
        self.assertIn("first doubt", self.amc._llama.system_prompt)
        self.assertIn("GPU chat cache was cleared", self.amc._llama.system_prompt)

    def test_event_e_clears_tutor_memory(self) -> None:
        saved: list = []

        class DC:
            def save_last_chats(self, turns):
                saved.extend(turns)

        amc = AMC(AMCConfig.mock(), data_controller=DC())
        try:
            amc.enter_event_t({"id": "s1"})
            amc.tutor_ask("q1")
            amc.enter_event_e(behavioral_level=2)
            self.assertEqual(len(amc._memory), 0)
            self.assertTrue(saved)
            st = amc.status()
            self.assertEqual(st.event, Event.E)
            self.assertEqual(st.role, LlamaRole.EVALUATOR)
        finally:
            amc.shutdown()

    def test_hint_requires_event_e(self) -> None:
        self.amc.enter_event_t({"id": "s1"})
        with self.assertRaises(RoleError):
            self.amc.evaluator_hint("2+2", "5")

    def test_ask_requires_event_t(self) -> None:
        self.amc.enter_event_e()
        with self.assertRaises(RoleError):
            self.amc.tutor_ask("hello")

    def test_evaluator_prompt_forbids_answer(self) -> None:
        prompt = evaluator_system_prompt(
            behavioral_level=2,
            question="2+2",
            student_answer="5",
            allow_approximate=False,
        )
        self.assertIn("Never state the correct answer", prompt)
        self.assertIn("exact match", prompt)
        self.assertIn("Behavioral gear level: 2", prompt)

    def test_tutor_prompt_embeds_section_json(self) -> None:
        prompt = tutor_system_prompt({"id": "s1", "text": "photosynthesis"})
        self.assertIn("photosynthesis", prompt)
        self.assertIn("role TUTOR", prompt)

    def test_talk_is_not_whisper(self) -> None:
        with self.assertRaises(TTSNotWiredError):
            self.amc.talk()

    def test_second_request_waits_for_first(self) -> None:
        llama = _SlowLlama(hold=0.25)
        whisper = MockWhisperBackend()
        amc = AMC(AMCConfig.mock(), llama=llama, whisper=whisper)
        try:
            amc.enter_event_t({"id": "s1"})
            results: list[str] = []

            def ask() -> None:
                results.append(amc.tutor_ask("slow"))

            t = threading.Thread(target=ask)
            t.start()
            self.assertTrue(llama.started.wait(timeout=2))
            # Whisper must not start until llama generate returns.
            amc.transcribe([0.1] * 8)
            t.join(timeout=2)
            self.assertEqual(llama.calls.count("unload"), 1)
            self.assertEqual(whisper.calls, ["load", "transcribe"])
            self.assertTrue(results)
            self.assertFalse(llama.is_loaded())
            self.assertTrue(whisper.is_loaded())
        finally:
            amc.shutdown()

    def test_never_two_models_loaded(self) -> None:
        loads: list[str] = []

        class L4(MockLlamaBackend):
            def load(self) -> None:
                super().load()
                loads.append("int4")
                if int8.is_loaded() or whisper.is_loaded() or tr.is_loaded():
                    loads.append("BOTH")

        class L8(MockLlamaBackend):
            def load(self) -> None:
                super().load()
                loads.append("int8")
                if int4.is_loaded() or whisper.is_loaded() or tr.is_loaded():
                    loads.append("BOTH")

        class W(MockWhisperBackend):
            def load(self) -> None:
                super().load()
                loads.append("whisper")
                if int4.is_loaded() or int8.is_loaded() or tr.is_loaded():
                    loads.append("BOTH")

        class T(MockTranslatorBackend):
            def load(self) -> None:
                super().load()
                loads.append("translate")
                if int4.is_loaded() or int8.is_loaded() or whisper.is_loaded():
                    loads.append("BOTH")

        int4 = L4("llama_int4")
        int8 = L8("llama_int8")
        whisper = W()
        tr = T()
        amc = AMC(
            AMCConfig.mock(),
            llama=int4,
            llama_int8=int8,
            whisper=whisper,
            translator=tr,
        )
        try:
            amc.enter_event_t({"id": "s1"})
            amc.transcribe([0.0])
            amc.translate("Hello.", source_lang="en", target_lang="hi")
            amc.enter_event_e()
            self.assertNotIn("BOTH", loads)
            self.assertEqual(loads, ["int4", "whisper", "translate", "int8"])
            self.assertFalse(int4.is_loaded())
            self.assertFalse(tr.is_loaded())
            self.assertTrue(int8.is_loaded())
        finally:
            amc.shutdown()

    def test_event_e_uses_int8_not_int4(self) -> None:
        self.amc.enter_event_t({"id": "s1"})
        self.assertTrue(self.amc._llama_int4.is_loaded())
        self.amc.enter_event_e(behavioral_level=2)
        self.assertFalse(self.amc._llama_int4.is_loaded())
        self.assertTrue(self.amc._llama_int8.is_loaded())
        self.assertEqual(self.amc.status().resident_model, ModelId.LLAMA_INT8)
        hint = self.amc.evaluator_hint("2+2", "5")
        self.assertTrue(hint)

    def test_leave_role_clears_kv_and_memory(self) -> None:
        self.amc.enter_event_t({"id": "s1"})
        self.amc.tutor_ask("q")
        snap = self.amc.leave_role()
        self.assertTrue(snap)
        self.assertEqual(len(self.amc._memory), 0)
        self.assertIsNone(self.amc.status().role)
        self.assertIn("finish_session", self.amc._llama.calls)

    def test_shutdown_unloads(self) -> None:
        self.amc.enter_event_t({"id": "s1"})
        self.amc.shutdown()
        self.assertFalse(self.amc._llama.is_loaded())
        self.assertFalse(self.amc._whisper.is_loaded())
        self.assertFalse(self.amc._translator.is_loaded())
        self.assertIsNone(self.amc.status().resident_model)

    def test_translate_unloads_llama_and_unloads_after(self) -> None:
        self.amc.enter_event_t({"id": "s1"})
        self.amc.tutor_ask("explain")
        out = self.amc.translate("Hello.", source_lang="en", target_lang="hi")
        self.assertEqual(out, "Hello.")
        self.assertFalse(self.amc._llama.is_loaded())
        self.assertFalse(self.amc._translator.is_loaded())
        self.assertIsNone(self.amc.status().resident_model)
        self.assertGreater(len(self.amc._memory), 0)

    def test_translate_both_directions_use_chunker(self) -> None:
        student = "Namaste. How are you?"
        to_en = self.amc.translate(student, source_lang="hi", target_lang="en")
        self.assertEqual(to_en, student)
        answer = "A fraction is a part of a whole. The top is the numerator."
        to_hi = self.amc.translate(answer, source_lang="en", target_lang="hi")
        self.assertEqual(to_hi, answer)
        self.assertFalse(self.amc._translator.is_loaded())

    def test_translate_empty_skips_gpu(self) -> None:
        out = self.amc.translate("   ", source_lang="en", target_lang="hi")
        self.assertEqual(out, "")
        self.assertFalse(self.amc._translator.is_loaded())
        self.assertEqual(self.amc._translator.calls, [])


class IRPathTests(unittest.TestCase):
    def test_missing_dir(self) -> None:
        with self.assertRaises(ModelPathError):
            _require_ir_dir("/no/such/ir", "Llama")

    def test_dir_only_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(ModelPathError) as ctx:
                _require_ir_dir(raw, "Llama 3.1 (llama_int8)")
            self.assertIn("incomplete", str(ctx.exception).lower())
            self.assertIn("openvino_model.bin", str(ctx.exception))

    def test_part_file_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "openvino_model.bin.part").write_bytes(b"partial")
            with self.assertRaises(ModelPathError) as ctx:
                _require_ir_dir(raw, "Llama 3.1 (llama_int8)")
            msg = str(ctx.exception)
            self.assertIn("openvino_model.bin.part", msg)
            self.assertIn("Incomplete download", msg)

    def test_complete_llama_ir_ok(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for name in (
                "openvino_model.xml",
                "openvino_model.bin",
                "openvino_tokenizer.xml",
                "openvino_tokenizer.bin",
            ):
                (root / name).write_bytes(b"x")
            self.assertEqual(_require_ir_dir(raw, "Llama"), root)


class TranslatorKvTests(unittest.TestCase):
    def test_default_translate_kv_is_one_gb(self) -> None:
        self.assertEqual(DEFAULT_KV_CACHE_GB_TRANSLATE, 1)
        self.assertEqual(AMCConfig().kv_cache_gb_translate, 1)


class GpuGuardTests(unittest.TestCase):
    def test_detects_out_of_resources(self) -> None:
        self.assertTrue(
            is_gpu_dead_error(
                RuntimeError("[GPU] clFinish, error code: -5 CL_OUT_OF_RESOURCES")
            )
        )

    def test_detects_wait_list(self) -> None:
        self.assertTrue(
            is_gpu_dead_error(
                RuntimeError("CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST")
            )
        )

    def test_detects_invalid_event(self) -> None:
        self.assertTrue(
            is_gpu_dead_error(RuntimeError("clWaitForEvents, error code: -58 CL_INVALID_EVENT"))
        )

    def test_ignores_normal_errors(self) -> None:
        self.assertFalse(is_gpu_dead_error(RuntimeError("missing IR")))

    def test_fp32_embeddings_are_unsafe(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "openvino_language_model.bin").write_bytes(b"x" * 100)
            xml = root / "openvino_text_embeddings_model.xml"
            xml.write_text('<net><layer precision="FP32"/></net>')
            (root / "openvino_text_embeddings_model.bin").write_bytes(b"y" * 100)
            (root / "openvino_vision_embeddings_model.xml").write_text(
                '<net><layer precision="U8"/></net>'
            )
            (root / "openvino_vision_embeddings_model.bin").write_bytes(b"v" * 100)
            reason = translator_unsafe_reason(root)
            self.assertIsNotNone(reason)
            self.assertIn("not INT8", reason or "")

    def test_vision_file_is_unsafe(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "openvino_text_embeddings_model.xml").write_text(
                '<net><layer precision="U8"/></net>'
            )
            vis = root / "openvino_vision_embeddings_model.bin"
            vis.write_bytes(b"v" * 20_000_000)
            (root / "openvino_vision_embeddings_model.xml").write_text(
                '<net><layer precision="FP32"/></net>'
            )
            reason = translator_unsafe_reason(root)
            self.assertIsNotNone(reason)
            self.assertIn("vision", reason or "")
            self.assertIn("SigLIP", reason or "")

    def test_missing_vision_xml_is_unsafe(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "openvino_language_model.bin").write_bytes(b"x" * 100)
            xml = root / "openvino_text_embeddings_model.xml"
            xml.write_text('<net><layer precision="U8"/></net>')
            (root / "openvino_text_embeddings_model.bin").write_bytes(b"y" * 100)
            reason = translator_unsafe_reason(root)
            self.assertIsNotNone(reason)
            self.assertIn("vision IR missing", reason or "")

    def test_large_int8_vision_is_unsafe(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "openvino_language_model.bin").write_bytes(b"x" * 100)
            (root / "openvino_text_embeddings_model.xml").write_text(
                '<net><layer precision="U8"/></net>'
            )
            (root / "openvino_text_embeddings_model.bin").write_bytes(b"y" * 100)
            (root / "openvino_vision_embeddings_model.xml").write_text(
                '<net><layer precision="U8"/></net>'
            )
            (root / "openvino_vision_embeddings_model.bin").write_bytes(
                b"v" * 20_000_000
            )
            reason = translator_unsafe_reason(root)
            self.assertIsNotNone(reason)
            self.assertIn("SigLIP", reason or "")

    def test_vision_stub_with_int8_embeddings_is_safe(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "openvino_language_model.bin").write_bytes(b"x" * 100)
            (root / "openvino_text_embeddings_model.xml").write_text(
                '<net><layer precision="U8"/></net>'
            )
            (root / "openvino_text_embeddings_model.bin").write_bytes(b"y" * 100)
            (root / "openvino_vision_embeddings_model.xml").write_text("<net/>")
            (root / "openvino_vision_embeddings_model.bin").write_bytes(b"v" * 100)
            self.assertIsNone(translator_unsafe_reason(root))


class ChunkerTests(unittest.TestCase):
    def test_prompt_limit_is_75_percent_of_2k(self) -> None:
        self.assertEqual(prompt_token_limit(2048, 0.75), 1536)

    def test_split_keeps_full_sentences(self) -> None:
        text = "Hello there. How are you? I am fine."
        parts = split_sentences(text)
        self.assertGreaterEqual(len(parts), 2)
        self.assertEqual("".join(parts), text)
        self.assertTrue(parts[0].strip().endswith("."))
        self.assertNotIn("How", parts[0])

    def test_does_not_split_mr_or_decimal(self) -> None:
        text = "Ask Mr. Shah about 3.14 and then stop."
        parts = split_sentences(text)
        self.assertEqual("".join(parts), text)
        self.assertEqual(len(parts), 1)

    def test_danda_is_a_sentence_end(self) -> None:
        text = "यह एक वाक्य है। दूसरा वाक्य है।"
        parts = split_sentences(text)
        self.assertEqual("".join(parts), text)
        self.assertGreaterEqual(len(parts), 2)

    def test_pack_cuts_after_sentence_under_limit(self) -> None:
        text = "One sentence here. Two sentence here. Three sentence here."
        chunks = pack_text(text, count=lambda s: len(s.split()), limit=6)
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(c.text for c in chunks), text)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text.split()), 6)
            stripped = chunk.text.strip()
            self.assertTrue(
                stripped.endswith(".") or chunk is chunks[-1],
                msg=chunk.text,
            )

    def test_chunk_for_translator_counts_full_prompt(self) -> None:
        def count(s: str) -> int:
            return len(s)

        text = "Alpha is first. Beta is second. Gamma is third. Delta is fourth."
        chunks = chunk_for_translator(
            text,
            source_lang="en",
            target_lang="hi",
            count=count,
            max_input_tokens=600,
            fill_ratio=0.75,
        )
        limit = prompt_token_limit(600, 0.75)
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(c.text for c in chunks), text)
        for chunk in chunks:
            prompt = format_translate_prompt(
                chunk.text, source_lang="en", target_lang="hi"
            )
            self.assertLessEqual(count(prompt), limit)

    def test_oversize_sentence_last_resort_does_not_drop(self) -> None:
        text = "word " * 40
        chunks = pack_text(text, count=lambda s: len(s.split()), limit=5)
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(c.text for c in chunks), text)

    def test_long_paragraph_is_chunked_not_dropped(self) -> None:
        sentences = [
            f"Sentence number {i} explains a piece of the fraction lesson."
            for i in range(1, 25)
        ]
        text = " ".join(sentences)
        chunks = chunk_for_translator(
            text,
            source_lang="en",
            target_lang="hi",
            count=len,
            max_input_tokens=600,
            fill_ratio=0.75,
        )
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(c.text for c in chunks), text)
        limit = prompt_token_limit(600, 0.75)
        for chunk in chunks:
            prompt = format_translate_prompt(
                chunk.text, source_lang="en", target_lang="hi"
            )
            self.assertLessEqual(len(prompt), limit)

    def test_one_short_sentence_is_a_single_chunk(self) -> None:
        text = "What is a fraction, in one short sentence?"
        chunks = chunk_for_translator(
            text,
            source_lang="en",
            target_lang="hi",
            count=lambda s: max(1, len(s) // 4),
            max_input_tokens=2048,
            fill_ratio=0.75,
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, text)


if __name__ == "__main__":
    unittest.main()
