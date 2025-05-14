import numpy as np
import pandas as pd
import random


class FacialAnalyzer:

    def __init__(self):
        pass

    def process_openface_data(self, openface_data):
        """
        Process OpenFace data and return jaw angle and head pitch metrics.
        Args:
            openface_data: DataFrame containing OpenFace analysis results
        Returns:
            Dictionary containing jaw angle and head pitch scores
        """
        try:
            if openface_data.empty:
                return {"error": "No data provided"}

            # Calculate metrics using OpenFace data
            jaw_angle, head_pitch = self.get_head_pose_openface(openface_data)

            return {"jaw_angle": jaw_angle, "head_pitch": head_pitch}
        except Exception as e:
            return {"error": str(e)}

    def get_head_pose_openface(self, openface_data):
        """
        Calculate head pose metrics from OpenFace data
        
        Args:
            openface_data: DataFrame containing OpenFace analysis results
            
        Returns:
            Tuple of (jaw_angle, head_pitch)
        """
        try:
            # Extract head pose values
            head_pitch = 0
            if ' pose_Rx' in openface_data.keys():
                # Convert from radians to degrees and normalize
                head_pitch = np.degrees(openface_data[' pose_Rx'])
            # For jaw angle, we can use a combination of AUs related to jaw movement
            # or use the pose_Ry (yaw) as an approximation
            jaw_angle = 100  # Default value

            if ' pose_Ry' in openface_data.keys():
                # Use yaw as a component of jaw angle
                yaw_degrees = np.degrees(abs(openface_data[' pose_Ry']))
                # Scale yaw to approximate jaw angle (100 is neutral)
                jaw_angle = 100 + yaw_degrees * 0.5

            # We can also incorporate AU25 (lips part) and AU26 (jaw drop) if available
            if ' AU25_r' in openface_data.keys(
            ) and ' AU26_r' in openface_data.keys():
                # These AUs can indicate jaw movement
                au25 = openface_data[' AU25_r'] / 5.0  # Normalize to 0-1
                au26 = openface_data[' AU26_r'] / 5.0  # Normalize to 0-1

                # Adjust jaw angle based on these AUs
                jaw_angle += (au25 + au26 * 2) * 10  # Scale appropriately

            return jaw_angle, head_pitch

        except Exception as e:
            print(f"Error calculating head pose: {e}")
            return 100, 0  # Default values on error


def calculate_section2(openface_results):
    analyzer = FacialAnalyzer()
    persistant_list = []
    out_focus_list = []
    out_structure_list = []
    out_risk_list = []

    # Check if we have valid data
    if not openface_results or isinstance(
            openface_results, dict) and 'error' in openface_results:
        return {"error": "No valid OpenFace data provided."}

    # Process each result (each result corresponds to one image/frame)
    df = pd.read_csv(openface_results)
    for i, result_data in df.iterrows():
        try:
            result = analyzer.process_openface_data(result_data)

            if 'error' in result:
                raise ValueError(f"Face detection failed: {result['error']}")

            # Extract Action Units (AUs) from OpenFace data
            au_values = {}
            for au in ['AU01', 'AU02', 'AU04', 'AU06']:
                au_col = f" {au}_r"  # OpenFace uses _r suffix for AU intensity
                if au_col in result_data.keys():
                    au_values[au] = result_data[au_col].mean(
                    ) / 5.0  # Normalize to 0-1 range
                else:
                    au_values[au] = 0.0  # Default if AU not available

            # Calculate metrics using the same formulas as before
            persistant = (1 - abs(result['jaw_angle'] - 100) / 40) * 0.3 + (
                1 - abs(result['head_pitch']) / 30) * 0.4 + (au_values.get(
                    'AU01', 0) + au_values.get('AU06', 0)) / 2 * 0.3
            persistant_list.append(make_score(persistant))

            out_focus = (1 - abs(result['head_pitch']) / 30) * 0.4 + (
                au_values.get('AU01', 0) + au_values.get('AU02', 0) +
                au_values.get('AU06', 0)) * 0.2
            out_focus_list.append(make_score(out_focus))

            out_structure = result['jaw_angle'] / 140 * 0.6 + (
                1 - abs(result['head_pitch']) / 30) * 0.4
            out_structure_list.append(make_score(out_structure))

            out_risk = (1 if result['head_pitch'] > 5 else 0) * 0.4 + (
                1 if result['jaw_angle'] > 120 else 0) * 0.3 + au_values.get(
                    'AU04', 0) * 0.3
            out_risk_list.append(make_score(out_risk))
        except Exception as e:
            # Log the error and continue with next result
            print(f"Error processing result {i}: {str(e)}")
            continue

    # Only proceed if we have processed at least one image successfully
    if len(persistant_list) == 0:
        return {"error": "Could not process any images"}

    try:
        select = np.argmax([
            persistant_list, out_focus_list, out_structure_list, out_risk_list
        ],
                           axis=1)
        return {
            "Persistant": {
                "balance": f"{np.mean(persistant_list)*100:.1f}%",
                "top_image": int(select[0])
            },
            "Focus": {
                "balance": f"{np.mean(out_focus_list)*100:.1f}%",
                "top_image": int(select[1])
            },
            "Structure": {
                "balance": f"{np.mean(out_structure_list)*100:.1f}%",
                "top_image": int(select[2])
            },
            "Risk": {
                "balance": f"{np.mean(out_risk_list)*100:.1f}%",
                "top_image": int(select[3])
            },
        }
    except Exception as e:
        return {"error": f"Failed to calculate final metrics: {str(e)}"}


def make_score(score):
    if score <= 0.3:
        score = 0.3 + 0.1 * score + random.random() * 0.07
    if score >= 0.93:
        score = 0.93 - 0.1 * (1 - score) - random.random() * 0.05
        print(f"Adjusted score: {score}")
    return score
