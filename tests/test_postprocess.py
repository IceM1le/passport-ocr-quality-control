import unittest
from datetime import date

import postprocess


class PassportValidityTests(unittest.TestCase):
    def test_passport_issued_at_14_is_expired_long_after_20(self):
        self.assertFalse(postprocess.valid_passport("01.01.1980", "01.01.1994"))

    def test_passport_issued_at_20_is_valid_before_45(self):
        self.assertTrue(postprocess.valid_passport("01.01.1990", "01.01.2010"))

    def test_passport_issued_at_45_is_indefinitely_valid(self):
        self.assertTrue(postprocess.valid_passport("01.01.1960", "01.01.2005"))

    def test_issue_date_in_future_is_invalid(self):
        self.assertFalse(postprocess.valid_passport("01.01.2000", "01.01.2027"))

    def test_under_14_issue_age_is_invalid(self):
        self.assertFalse(postprocess.valid_passport("01.01.2010", "01.01.2023"))


class MrzTests(unittest.TestCase):
    def test_birth_date_check_digit(self):
        self.assertEqual(postprocess.mrz_check_digit("040416"), "9")

    def test_empirical_name_mapping(self):
        self.assertEqual(postprocess.to_mrz_name("МУТЯНСКИЙ"), "MUT8NSKIQ")

    def test_unknown_or_empty_name_is_not_transliterated(self):
        self.assertIsNone(postprocess.to_mrz_name(""))


class GeometryTests(unittest.TestCase):
    def test_tall_field_with_one_line_is_suspicious(self):
        obj = {"box": [100, 200, 400, 64], "lines": [{"text": "ОТДЕЛ МВД", "box": [102, 202, 390, 29]}]}
        self.assertTrue(postprocess.suspicious_long_field_geometry(obj))

    def test_normal_single_line_field_is_not_suspicious(self):
        obj = {"box": [100, 200, 400, 32], "lines": [{"text": "ИВАНОВ", "box": [102, 202, 390, 29]}]}
        self.assertFalse(postprocess.suspicious_long_field_geometry(obj))


if __name__ == "__main__":
    unittest.main()
