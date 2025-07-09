"""
Evaluator for the function minimization example
"""

import subprocess
import numpy as np
import time
import traceback
import re
import json
import shutil
import tempfile
import os


def get_ts_result(program_path):
    # 无论后缀如何，都读取内容写入临时 .ts 文件
    with open(program_path, 'r') as f:
        code = f.read()
    with tempfile.NamedTemporaryFile(suffix='.ts', delete=False, mode='w') as tmpf:
        tmpf.write(code)
        ts_path = tmpf.name
    try:
        result = subprocess.run([
            "npx", "ts-node", "--compiler-options", '{"module":"CommonJS"}', ts_path
        ], capture_output=True, text=True, timeout=10)
        output = result.stdout.strip()
        print("TS stdout:", output)
        print("TS stderr:", result.stderr)
        
        # Check if there's a TypeScript compilation error
        if "TSError" in result.stderr or "Unable to compile TypeScript" in result.stderr:
            print("TypeScript compilation failed, retrying with shell=True...")
            result2 = subprocess.run(
                f"npx ts-node --compiler-options '{{\"module\":\"CommonJS\"}}' {ts_path}",
                shell=True, capture_output=True, text=True, timeout=10
            )
            output2 = result2.stdout.strip()
            print("TS shell=True stdout:", output2)
            print("TS shell=True stderr:", result2.stderr)
            if not output2 or "TSError" in result2.stderr or "Unable to compile TypeScript" in result2.stderr:
                # Extract the main error message for cleaner output
                error_lines = result2.stderr.split('\n')
                main_error = ""
                for line in error_lines:
                    if "error TS" in line:
                        main_error = line.strip()
                        break
                if not main_error:
                    main_error = "TypeScript compilation failed"
                raise ValueError(f"TypeScript compilation failed: {main_error}")
            output = output2
        elif not output:
            print("Output is empty, retrying with shell=True...")
            result2 = subprocess.run(
                f"npx ts-node --compiler-options '{{\"module\":\"CommonJS\"}}' {ts_path}",
                shell=True, capture_output=True, text=True, timeout=10
            )
            output2 = result2.stdout.strip()
            print("TS shell=True stdout:", output2)
            print("TS shell=True stderr:", result2.stderr)
            if not output2:
                raise ValueError(f"Unrecognized output: {output2}\nStderr: {result2.stderr}")
            output = output2
        # 1. 尝试 JSON
        try:
            data = json.loads(output)
            if isinstance(data, dict) and all(k in data for k in ("x", "y", "value")):
                return (data["x"], data["y"], data["value"])
            elif isinstance(data, dict) and all(k in data for k in ("x", "y")):
                return (data["x"], data["y"])
            elif isinstance(data, list) and len(data) in (2, 3):
                return tuple(data)
        except Exception:
            pass
        # 2. 尝试正则，支持 'Found minimum at (x, y) with value z' 格式
        m = re.search(r"Found minimum at \(([-\d\.]+), ([-\d\.]+)\) with value ([-\d\.eE]+)", output)
        if m:
            return (float(m.group(1)), float(m.group(2)), float(m.group(3)))
        # 3. 尝试只提取 x, y
        m2 = re.search(r"\(([-\d\.]+), ([-\d\.]+)\)", output)
        if m2:
            return (float(m2.group(1)), float(m2.group(2)))
        raise ValueError(f"Unrecognized output: {output}\nStderr: {result.stderr}")
    finally:
        if os.path.exists(ts_path):
            os.remove(ts_path)


def safe_float(value):
    """Convert a value to float safely"""
    try:
        return float(value)
    except (TypeError, ValueError):
        print(f"Warning: Could not convert {value} of type {type(value)} to float")
        return 0.0


def evaluate(program_path):
    """
    Evaluate the program by running it multiple times and checking how close
    it gets to the known global minimum.

    Args:
        program_path: Path to the program file

    Returns:
        Dictionary of metrics
    """
    # Known global minimum (approximate)
    GLOBAL_MIN_X = -1.704
    GLOBAL_MIN_Y = 0.678
    GLOBAL_MIN_VALUE = -1.519

    try:
        # Run multiple trials
        num_trials = 10
        x_values = []
        y_values = []
        values = []
        distances = []
        times = []
        success_count = 0

        for trial in range(num_trials):
            try:
                start_time = time.time()
                result = get_ts_result(program_path)
                # 保留 tuple 判断和 value 自动补算逻辑
                if isinstance(result, tuple):
                    if len(result) == 3:
                        x, y, value = result
                    elif len(result) == 2:
                        x, y = result
                        value = np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20
                        print(f"Trial {trial}: Got 2 values, calculated function value: {value}")
                    else:
                        print(f"Trial {trial}: Invalid result format, expected tuple of 2 or 3 values but got {len(result)}")
                        continue
                else:
                    print(f"Trial {trial}: Invalid result format, expected tuple but got {type(result)}")
                    continue
                end_time = time.time()

                # Ensure all values are float
                x = safe_float(x)
                y = safe_float(y)
                value = safe_float(value)

                # Check if the result is valid (not NaN or infinite)
                if (
                    np.isnan(x)
                    or np.isnan(y)
                    or np.isnan(value)
                    or np.isinf(x)
                    or np.isinf(y)
                    or np.isinf(value)
                ):
                    print(f"Trial {trial}: Invalid result, got x={x}, y={y}, value={value}")
                    continue

                # Calculate metrics
                x_diff = x - GLOBAL_MIN_X
                y_diff = y - GLOBAL_MIN_Y
                distance_to_global = np.sqrt(x_diff**2 + y_diff**2)

                x_values.append(x)
                y_values.append(y)
                values.append(value)
                distances.append(distance_to_global)
                times.append(end_time - start_time)
                success_count += 1

            except Exception as e:
                print(f"Trial {trial}: Error - {str(e)}")
                print(traceback.format_exc())
                continue

        # If all trials failed, return zero scores
        if success_count == 0:
            return {
                "value_score": 0.0,
                "distance_score": 0.0,
                "speed_score": 0.0,
                "combined_score": 0.0,
                "error": "All trials failed",
            }

        # Calculate metrics
        avg_value = float(np.mean(values))
        avg_distance = float(np.mean(distances))
        avg_time = float(np.mean(times)) if times else 1.0

        # Convert to scores (higher is better)
        value_score = float(1.0 / (1.0 + abs(avg_value - GLOBAL_MIN_VALUE)))  # Normalize and invert
        distance_score = float(1.0 / (1.0 + avg_distance))
        speed_score = float(1.0 / avg_time) if avg_time > 0 else 0.0

        # calculate standard deviation scores
        x_std_score = float(1.0 / (1.0 + np.std(x_values)))
        y_std_score = float(1.0 / (1.0 + np.std(x_values)))
        standard_deviation_score = (x_std_score + y_std_score) / 2.0

        # Normalize speed score (so it doesn't dominate)
        speed_score = float(min(speed_score, 10.0) / 10.0)

        # Add reliability score based on success rate
        reliability_score = float(success_count / num_trials)

        # Calculate a single combined score that prioritizes finding good solutions
        # over secondary metrics like speed and reliability
        # Value and distance scores (quality of solution) get 90% of the weight
        # Speed and reliability get only 10% combined
        combined_score = float(
            0.35 * value_score
            + 0.35 * distance_score
            + standard_deviation_score * 0.20
            + 0.05 * speed_score
            + 0.05 * reliability_score
        )

        # Also compute an "overall" score that will be the primary metric for selection
        # This adds a bonus for finding solutions close to the global minimum
        # and heavily penalizes solutions that aren't finding the right region
        if distance_to_global < 1.0:  # Very close to the correct solution
            solution_quality = 1.0
        elif distance_to_global < 3.0:  # In the right region
            solution_quality = 0.5
        else:  # Not finding the right region
            solution_quality = 0.1

        # Overall score is dominated by solution quality but also factors in the combined score
        overall_score = 0.8 * solution_quality + 0.2 * combined_score

        return {
            "value_score": value_score,
            "distance_score": distance_score,
            "standard_deviation_score": standard_deviation_score,
            "speed_score": speed_score,
            "reliability_score": reliability_score,
            "combined_score": combined_score,
            "overall_score": overall_score,  # This will be the primary selection metric
            "success_rate": reliability_score,
        }
    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        print(traceback.format_exc())
        return {
            "value_score": 0.0,
            "distance_score": 0.0,
            "speed_score": 0.0,
            "combined_score": 0.0,
            "error": str(e),
        }


def evaluate_stage1(program_path):
    """First stage evaluation with fewer trials"""
    # Known global minimum (approximate)
    GLOBAL_MIN_X = float(-1.704)
    GLOBAL_MIN_Y = float(0.678)
    GLOBAL_MIN_VALUE = float(-1.519)

    try:
        # Quick check to see if the program runs without errors
        start_time = time.time()
        result = get_ts_result(program_path)
        if isinstance(result, tuple):
            if len(result) == 3:
                x, y, value = result
            elif len(result) == 2:
                x, y = result
                value = np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20
                print(f"Stage1: Got 2 values, calculated function value: {value}")
            else:
                print(f"Stage1: Invalid result format, expected tuple of 2 or 3 values but got {len(result)}")
                return {"runs_successfully": 0.0, "error": "Invalid result format"}
        else:
            print(f"Stage1: Invalid result format, expected tuple but got {type(result)}")
            return {"runs_successfully": 0.0, "error": "Invalid result format"}
        end_time = time.time()

        # Ensure all values are float
        x = safe_float(x)
        y = safe_float(y)
        value = safe_float(value)

        # Check if the result is valid
        if (
            np.isnan(x)
            or np.isnan(y)
            or np.isnan(value)
            or np.isinf(x)
            or np.isinf(y)
            or np.isinf(value)
        ):
            return {"runs_successfully": 0.5, "error": "Invalid result values"}

        # Calculate distance safely
        x_diff = float(x) - GLOBAL_MIN_X
        y_diff = float(y) - GLOBAL_MIN_Y
        distance = float(np.sqrt(x_diff**2 + y_diff**2))

        # Calculate value-based score
        value_score = float(1.0 / (1.0 + abs(value - GLOBAL_MIN_VALUE)))
        distance_score = float(1.0 / (1.0 + distance))

        # Calculate solution quality metric
        if distance < 1.0:  # Very close to the correct solution
            solution_quality = 1.0
        elif distance < 3.0:  # In the right region
            solution_quality = 0.5
        else:  # Not finding the right region
            solution_quality = 0.1

        # Basic metrics with overall score
        return {
            "runs_successfully": 1.0,
            "value_score": value_score,
            "distance_score": distance_score,
            "overall_score": solution_quality,  # This becomes a strong guiding metric
        }
    except Exception as e:
        print(f"Stage 1 evaluation failed: {e}")
        print(traceback.format_exc())
        return {"runs_successfully": 0.0, "error": str(e)}


def evaluate_stage2(program_path):
    """Second stage evaluation with more thorough testing"""
    # Full evaluation as in the main evaluate function
    return evaluate(program_path)
