from __future__ import annotations

import random
import unittest

from athena.amc import AMC, AMCConfig
from athena.amc.backends import MockTranslatorBackend
from athena.event_g import EventG, plan_hops
from athena.noise import inject_word_noise, word_error_rate


class PlanTests(unittest.TestCase):
    def test_english_keyboard_other_output(self) -> None:
        # scenario 1.1
        self.assertEqual(
            plan_hops(speech=False, speak=False, source_lang="en", target_lang="hi"),
            ["llama", "translate_out"],
        )

    def test_type_in_selected_language(self) -> None:
        # scenario 1.2
        self.assertEqual(
            plan_hops(speech=False, speak=False, source_lang="hi", target_lang="hi"),
            ["translate_in", "llama", "translate_out"],
        )

    def test_speech(self) -> None:
        self.assertEqual(
            plan_hops(speech=True, speak=False, source_lang="hi", target_lang="hi"),
            ["stt", "translate_in", "llama", "translate_out"],
        )

    def test_speech_plus_speak_adds_tts_only(self) -> None:
        # scenario 2 + 3: do not rerun the pipeline
        self.assertEqual(
            plan_hops(speech=True, speak=True, source_lang="hi", target_lang="hi"),
            ["stt", "translate_in", "llama", "translate_out", "tts"],
        )

    def test_english_speech_skips_translator(self) -> None:
        self.assertEqual(
            plan_hops(speech=True, speak=False, source_lang="en", target_lang="en"),
            ["stt", "llama"],
        )


class EventGMockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.amc = AMC(AMCConfig.mock())
        self.g = EventG(self.amc)

    def tearDown(self) -> None:
        self.amc.shutdown()

    def test_english_only_skips_translator(self) -> None:
        tr = self.amc._translator
        assert isinstance(tr, MockTranslatorBackend)
        out = self.g.run(text="what is a fraction?", source_lang="en", target_lang="en")
        self.assertEqual(out.plan, ["llama"])
        self.assertEqual(tr.generate_count, 0)
        self.assertIn("what is a fraction?", out.english_question)
        self.assertFalse(self.amc._translator.is_loaded())

    def test_roundtrip_calls_translate_twice_and_unloads(self) -> None:
        tr = self.amc._translator
        assert isinstance(tr, MockTranslatorBackend)
        out = self.g.run(
            text="what is a fraction?", source_lang="hi", target_lang="hi"
        )
        self.assertEqual(out.plan, ["translate_in", "llama", "translate_out"])
        self.assertEqual(tr.generate_count, 2)
        self.assertEqual([h.name for h in out.hops], out.plan)
        self.assertFalse(tr.is_loaded())
        # last hop is translator: exclusive slot unloads Llama too
        self.assertFalse(self.amc._llama.is_loaded())

    def test_speech_uses_whisper_then_translate_in(self) -> None:
        out = self.g.run(
            pcm=[0.0] * 160, speech=True, source_lang="hi", target_lang="en"
        )
        self.assertEqual(out.plan, ["stt", "translate_in", "llama"])
        self.assertEqual(out.hops[0].name, "stt")
        self.assertEqual(out.english_question, "what does this paragraph mean")
        self.assertFalse(self.amc._whisper.is_loaded())

    def test_speak_runs_tts_and_unloads(self) -> None:
        out = self.g.run(
            text="what is a fraction?",
            source_lang="en",
            target_lang="hi",
            speak=True,
        )
        self.assertIn("tts", out.plan)
        self.assertFalse(out.tts_pending)
        self.assertEqual(out.hops[-1].name, "tts")
        self.assertIn("<pcm", out.hops[-1].output_text)
        self.assertFalse(self.amc._tts.is_loaded())
        self.assertIsNone(self.amc.status().resident_model)


class NoiseTests(unittest.TestCase):
    def test_wer_zero_on_identity(self) -> None:
        s = "A fraction is a part of a whole."
        self.assertEqual(word_error_rate(s, s), 0.0)

    def test_more_hops_raise_wer(self) -> None:
        gold = (
            "What is a fraction of a whole. "
            "The numerator is the top number."
        )
        rng = random.Random(0)
        rate = 0.25
        one = inject_word_noise(gold, rate, rng)
        rng = random.Random(0)
        stacked = gold
        for _ in range(4):
            stacked = inject_word_noise(stacked, rate, rng)
        self.assertGreater(
            word_error_rate(gold, stacked), word_error_rate(gold, one)
        )


if __name__ == "__main__":
    unittest.main()
