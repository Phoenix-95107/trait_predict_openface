import numpy as np
import pandas as pd
import random


class FacialAnalyzer:

    def __init__(self):
        # No need for MediaPipe initialization anymore
        pass

    def process_openface_data(self, openface_data):
        """
        Process OpenFace data and return gaze and iris metrics.
        Args:
            openface_data: DataFrame containing OpenFace analysis results
        Returns:
            Dictionary containing gaze direction and iris ratio metrics
        """
        try:
            if openface_data.empty:
                return {"error": "No data provided"}
            # Calculate metrics using OpenFace data
            gaze_direction, iris_ratio = self.calculate_gaze_iris_openface(
                openface_data)
            return {"iris_ratio": iris_ratio, "gaze_direction": gaze_direction}
        except Exception as e:
            return {"error": str(e)}

    def calculate_gaze_iris_openface(self, openface_data):
        """
        Calculate gaze direction and iris ratio using OpenFace data.
        
        Args:
            openface_data: DataFrame containing OpenFace analysis results
            
        Returns:
            Tuple of (gaze_direction, iris_ratio)
        """
        try:
            # Extract gaze direction from OpenFace data
            # OpenFace provides gaze direction vectors for both eyes
            gaze_direction = 1  # Default to center (1)

            # Check if we have gaze direction data
            if all(col in openface_data.keys() for col in
                   [' gaze_0_x', ' gaze_0_y', ' gaze_1_x', ' gaze_1_y']):
                # Extract gaze vectors for both eyes
                left_gaze_x = openface_data[' gaze_0_x']
                left_gaze_y = openface_data[' gaze_0_y']
                right_gaze_x = openface_data[' gaze_1_x']
                right_gaze_y = openface_data[' gaze_1_y']

                # Average the gaze vectors from both eyes
                avg_gaze_x = (left_gaze_x + right_gaze_x) / 2
                avg_gaze_y = (left_gaze_y + right_gaze_y) / 2

                # Determine if gaze is centered
                # OpenFace gaze vectors are normalized, so we can use thresholds directly
                if abs(avg_gaze_x) <= 0.2 and abs(avg_gaze_y) <= 0.2:
                    gaze_direction = 1  # Center
                else:
                    gaze_direction = 0  # Not center

            # Calculate iris ratio
            # OpenFace doesn't directly provide iris measurements, but we can estimate
            # from eye openness and other features

            # Default value
            iris_ratio = 0.0

            # If we have eye aspect ratio or AU45 (blink), we can estimate
            if ' AU45_r' in openface_data.keys():
                # AU45 is blink - higher values mean more closed eyes
                blink_value = openface_data[' AU45_r']
                # Convert blink to openness (invert and normalize)
                eye_openness = 1.0 - min(
                    blink_value / 5.0,
                    1.0)  # OpenFace AU intensities are typically in 0-5 range

                # Estimate iris ratio from eye openness
                # This is a simplified model - actual clinical models would be more complex
                iris_ratio = eye_openness * 0.5  # Scale to a reasonable range

            # Alternative approach using AUs related to eye opening
            elif any(au in openface_data.keys()
                     for au in [' AU01_r', ' AU02_r', ' AU05_r']):
                # These AUs are related to eye opening (brow raiser, lid raiser)
                au_values = []
                for au in [' AU01_r', ' AU02_r', ' AU05_r']:
                    if au in openface_data.keys():
                        au_values.append(openface_data[au])

                if au_values:
                    # Average the relevant AUs and normalize
                    avg_au = sum(au_values) / len(au_values)
                    normalized_au = min(
                        avg_au / 5.0, 1.0
                    )  # OpenFace AU intensities are typically in 0-5 range

                    # Estimate iris ratio
                    iris_ratio = normalized_au * 0.5  # Scale to a reasonable range

            # If we have neither, use a default value
            else:
                iris_ratio = 0.0

            return gaze_direction, iris_ratio

        except Exception as e:
            print(f"Error calculating gaze and iris metrics: {e}")
            return 1, 0.0  # Default values on error


def calculate_section1(openface_results):
    analyzer = FacialAnalyzer()
    trust = []
    openness = []
    Empathy = []
    ConflictAvoid = []
    # Check if we have valid data
    if not openface_results or isinstance(
            openface_results, dict) and 'error' in openface_results:
        return {"error": "No valid OpenFace data provided."}

    # Process each result (each result corresponds to one image/frame)
    df = pd.read_csv(openface_results)
    for i, data in df.iterrows():
        try:
            result = analyzer.process_openface_data(data)
            if 'error' in result:
                print(f"Error processing data : {result['error']}")
                continue

            # Extract Action Units (AUs) from OpenFace data
            au_values = {}
            for au in ['AU01', 'AU06', 'AU12']:
                au_col = f" {au}_r"  # OpenFace uses _r suffix for AU intensity
                if au_col in data.keys():
                    au_values[au] = data[au_col].mean(
                    ) / 5.0  # Normalize to 0-1 range
                else:
                    au_values[au] = 0.0  # Default if AU not available

            # Calculate iris score
            iris_score = min(max((result['iris_ratio'] + 1) / 2, 0), 1)

            # Calculate metrics using the same formulas as before
            trust.append(
                make_score((au_values.get('AU06', 0) +
                            au_values.get('AU12', 0)) * 0.2 +
                           result['gaze_direction'] * 0.4 + iris_score * 0.2))

            openness.append(
                make_score(
                    au_values.get('AU12', 0) * 0.5 +
                    au_values.get('AU06', 0) * 0.3 +
                    result['gaze_direction'] * 0.2))

            Empathy.append(
                make_score(
                    au_values.get('AU01', 0) * 0.5 +
                    au_values.get('AU06', 0) * 0.3 +
                    result['gaze_direction'] * 0.2))

            ConflictAvoid.append(
                make_score((1 - au_values.get('AU12', 0)) * 0.5 +
                           (1 - au_values.get('AU06', 0)) * 0.3 +
                           result['gaze_direction'] * 0.2))
        except Exception as e:
            print(f"Error processing result: {str(e)}")
            # Continue with next result instead of failing completely
            continue
    try:
        # Find the top image for each metric
        select = np.argmax([trust, openness, Empathy, ConflictAvoid], axis=1)

        return {
            "trust": {
                "balance": f"{np.mean(trust)*100:.1f}%",
                "top_image": int(select[0])
            },
            "openness": {
                "balance": f"{np.mean(openness)*100:.1f}%",
                "top_image": int(select[1])
            },
            "Empathy": {
                "balance": f"{np.mean(Empathy)*100:.1f}%",
                "top_image": int(select[2])
            },
            "ConflictAvoid": {
                "balance": f"{np.mean(ConflictAvoid)*100:.1f}%",
                "top_image": int(select[3])
            },
        }
    except Exception as e:
        return {"error": f"Failed to calculate final metrics: {str(e)}"}


def make_score(score):
    if score <= 0.3:
        score = 0.3 + 0.1 * score + random.random() * 0.07
    if score >= 0.93:
        score = 0.93 - 0.1 * (1 - score) - random.random() * 0.03
        print(f"Adjusted score: {score}")
    return score
