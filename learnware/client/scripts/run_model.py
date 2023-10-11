import sys
import pickle
import argparse
from learnware.utils import get_module_by_module_path


def run_model(model_path, input_path, output_path):
    output_results = {"status": "success"}

    try:
        with open(model_path, "rb") as model_file:
            model_config = pickle.load(file=model_file)

        model_module = get_module_by_module_path(model_config["module_path"])
        cls = getattr(model_module, model_config["class_name"])
        setattr(sys.modules["__main__"], model_config["class_name"], cls)
        model = cls(**model_config.get("kwargs", {}))

        output_results["metadata"] = {
            "input_shape": model.input_shape,
            "output_shape": model.output_shape,
        }

        if input_path is not None:
            with open(input_path, "rb") as input_file:
                input_args = pickle.load(input_file)
            output_array = getattr(model, input_args.get("method", "predict"))(**input_args.get("kargs", {}))
            output_results[input_args.get("method", "predict")] = output_array

    except Exception as e:
        output_results["status"] = "fail"
        output_results["error_info"] = e
        raise e

    with open(output_path, "wb") as output_file:
        pickle.dump(output_results, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="path of model config")
    parser.add_argument("--input-path", type=str, required=False, help="path of input array")
    parser.add_argument("--output-path", type=str, required=True, help="path of output array")

    args = parser.parse_args()

    model_path = args.model_path
    input_path = args.input_path
    output_path = args.output_path

    run_model(model_path, input_path, output_path)
