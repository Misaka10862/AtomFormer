import unittest
from unittest import mock

from condmolgen.representation import assemble_with_representation, get_representation_spec
from condmolgen.decoration_data import DecorationDataset, build_decoration_tokenizer


class RepresentationSpecTests(unittest.TestCase):
    def test_rooted_and_plain_map_to_distinct_targets_and_assembly(self):
        rooted = get_representation_spec("rooted")
        plain = get_representation_spec("plain")
        self.assertEqual(rooted.target_key, "rooted_decoration_smiles")
        self.assertEqual(plain.target_key, "plain_decoration_smiles")
        self.assertEqual(rooted.assembly_mode, "assemble_single_attachment_rooted")
        self.assertEqual(plain.assembly_mode, "assemble_single_attachment_plain")

    def test_dataset_switches_target_without_rerooting(self):
        records = [
            {
                "scaffold_smiles": "[*:1]CC",
                "rooted_decoration_smiles": "NCC",
                "plain_decoration_smiles": "CCN",
                "cond_z": [6, 6],
                "cond_pos": [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
                "anchor_z": 7,
            }
        ]
        tokenizer = build_decoration_tokenizer(records, target_key="plain_decoration_smiles")
        dataset = DecorationDataset(
            records,
            tokenizer,
            target_key="plain_decoration_smiles",
        )
        self.assertEqual(dataset[0].target_decoration_smiles, "CCN")

    def test_assembly_dispatches_to_selected_routine(self):
        with mock.patch(
            "condmolgen.assembly.assemble_single_attachment_rooted",
            return_value="rooted-result",
        ) as rooted_mock:
            self.assertEqual(assemble_with_representation("rooted", "S", "D"), "rooted-result")
            rooted_mock.assert_called_once_with("S", "D")
        with mock.patch(
            "condmolgen.assembly.assemble_single_attachment_plain",
            return_value="plain-result",
        ) as plain_mock:
            self.assertEqual(assemble_with_representation("plain", "S", "D"), "plain-result")
            plain_mock.assert_called_once_with("S", "D")

    def test_known_assembly_canonical_outputs(self):
        try:
            from condmolgen.assembly import assemble_single_attachment_plain, assemble_single_attachment_rooted
        except RuntimeError:
            self.skipTest("RDKit is unavailable")
        try:
            rooted_result = assemble_single_attachment_rooted("[*:1]CC", "NCC")
            plain_result = assemble_single_attachment_plain("[*:1]CC", "CCN")
        except RuntimeError:
            self.skipTest("RDKit is unavailable")
        self.assertEqual(rooted_result, "CCNCC")
        self.assertEqual(plain_result, "CCCCN")


if __name__ == "__main__":
    unittest.main()
