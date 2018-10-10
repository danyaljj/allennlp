import json


class Prerequisit_Skills():

    skills_map = {
        0 : "Object  Tracking",
        1: "Mathematical Reasoning",
        2: "Coreference",
        3: "Logical Reasoning",
        4: "Analogy",
        5: "Causality",
        6: "Spatio-temporal",
        7: "Ellipsis (implicit info)",
        8: "Bridging",
        9: "Elaboration",
        10: "Meta-knowledge",
        11: "Schematic clauses",
        12: "Punctuation",
        13: "No skill",
        14: "Nonsense",
    }

    folder = "/Users/daniel/ideaProjects/allennlp/QA_datasets/prerequisit_data/"
    # {"original_id": "mc160.dev.0",
    #  "annotations": [
    #      {"skills": [13], "sents_indices": [[0, 5]], "skill_count": 0, "nonsense": false},
    #      {"skills": [0, 9, 2], "sents_indices": [[31, 43], [5, 10]], "skill_count": 3, "nonsense": false},
    #      {"skills": [11], "sents_indices": [[21, 31]], "skill_count": 1, "nonsense": false},
    #      {"skills": [0], "sents_indices": [[31, 43]], "skill_count": 1, "nonsense": false}
    #     ],
    #  "id": "mctest_000"}
    def read_prereq_file(self, file_name):
        all_anno = {}
        file_name = self.folder + file_name
        with open(file_name) as file:
            dataset_json = json.load(file)
            for a in dataset_json:
                original_id = a["original_id"]
                annotations = a["annotations"]
                # new_annotations = []
                for i, aa in enumerate(annotations):
                    # print(aa["skills"])
                    skills = [self.skills_map[x] for x in aa["skills"]]
                    all_anno[original_id + "-" + str(i + 1)] = skills

        return all_anno

preq = Prerequisit_Skills()
print(len(preq.read_prereq_file("mctest.json")))