import pathlib


class Paths:
    base_path = str(pathlib.Path(__file__).parent.resolve()) + "/data/{dir}"
    identifier_path = "/{env}/{set_name}"

    TEST_SET = base_path.format(dir="test_sets") + identifier_path + ".YAML"
    NAMED_POLICY = base_path.format(dir="policies") + identifier_path
    EVAL = base_path.format(dir="evaluation") + identifier_path + ".YAML"


class PolicyIDs:
    MAIN_POLICY_ID = "MAIN_POLICY"
    MAIN_POLICY_COPY_ID = "MAIN_POLICY_COPY"  # An older version of the main policy