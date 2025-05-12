import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

class FacialAnalyzer:
    def __init__(self):
        pass

    def process_openface_data(self, openface_data):
        """
        Process single-row OpenFace data and return eye openness and facial symmetry metrics.
        Args:
            openface_data: DataFrame with one row of OpenFace results
        Returns:
            Dictionary with eye openness and facial symmetry scores
        """
        try:
            eye_openness = self.calculate_eye_openness(openface_data)
            facial_symmetry = self.calculate_facial_symmetry(openface_data)
            return {
                "eye_openness": eye_openness,
                "facial_symmetry": facial_symmetry
            }
        except Exception as e:
            return {"error": str(e)}

    def calculate_eye_openness(self, row):
        """
        Estimate eye openness using AU45_r or other AUs from a single OpenFace row.
        """
        try:
            col_names = row.keys()
            # # Try AU45_r (inverse of eye openness)
            if ' AU45_r' in col_names:
                blink_val = row[' AU45_r']
                normalized_openness = 1.0 - min(blink_val / 5.0, 1.0)
                return normalized_openness

            # Alternative: use AU01, AU02, AU05
            au_cols = [c for c in [' AU01_r', ' AU02_r', ' AU05_r']]
            au_values = [row[c] for c in au_cols]
            normalized_openness = min(np.mean(au_values) / 5.0, 1.0)
            return normalized_openness
        except Exception as e:
            print(f"Eye openness error: {e}")
            return 0.5

    def calculate_facial_symmetry(self, row):
        """
        Calculate symmetry score from facial landmarks in a single row.
        """
        try:
           # Extract landmark x and y coordinate columns
            x_cols = [col for col in row.keys() if col.startswith(" x_")]
            y_cols = [col for col in row.keys() if col.startswith(" y_")]

            if not x_cols or not y_cols:
                return 0.5  # Default if landmarks missing

            x_values = [row[col] for col in x_cols]
            midline_x = np.mean(x_values)

            asymmetry_scores = []

            for i in range(len(x_cols) // 2):
                left_idx = i
                right_idx = len(x_cols) - 1 - i
                if left_idx >= right_idx:
                    break

                left_x = row[x_cols[left_idx]]
                right_x = row[x_cols[right_idx]]

                left_y = row[y_cols[left_idx]]
                right_y = row[y_cols[right_idx]]

                dist_diff = abs(abs(left_x - midline_x) - abs(right_x - midline_x))
                y_diff = abs(left_y - right_y)

                pair_asymmetry = dist_diff + 0.5 * y_diff
                asymmetry_scores.append(pair_asymmetry)

            if not asymmetry_scores:
                return 0.5

            avg_asymmetry = np.mean(asymmetry_scores)
            symmetry_score = max(0, 1 - (avg_asymmetry * 10))  # Scaled for simplicity

            return symmetry_score

        except Exception as e:
            print(f"Symmetry error: {e}")
            return 0.5

# Updated function to work with OpenFace data
def calculate_section3(openface_results):
    analyzer = FacialAnalyzer()
    ideation_list = []
    openness_list = []
    originalty_list = []
    attention_list = []
    
    df = pd.read_csv(openface_results)
    # Process each result (each result corresponds to one image/frame)
    for i, result_data in df.iterrows():
        try:
            metrics = analyzer.process_openface_data(result_data)
            if 'error' in metrics:
                print(f"Error processing data {i}: {metrics['error']}")
                continue
            
            # Extract Action Units (AUs) from OpenFace data
            au_values = {}
            for au in ['AU01', 'AU02', 'AU04', 'AU06', 'AU12']:
                au_col = f" {au}_r"  # OpenFace uses _r suffix for AU intensity
                if au_col in result_data.keys():
                    au_values[au] = result_data[au_col].mean() / 5.0  # Normalize to 0-1 range
                else:
                    au_values[au] = 0.0  # Default if AU not available
            
            # Calculate micro expression score
            micro_expr_score = 1.0 if any([au_values.get(au, 0) > 0.9 for au in ['AU01', 'AU02', 'AU04', 'AU12']]) else 0.0
            
            # Calculate metrics using the same formulas as before
            ideation = (au_values.get('AU12', 0) + au_values.get('AU06', 0))*0.2 + metrics['eye_openness']*0.3 + micro_expr_score*0.3
            ideation_list.append(ideation)
            
            openness = (au_values.get('AU02', 0) + au_values.get('AU05', 0))*0.3 + metrics['facial_symmetry']*0.4
            openness_list.append(openness)
            
            originalty = (1 - metrics['facial_symmetry'])*0.4 + au_values.get('AU12', 0)*0.3 + au_values.get('AU01', 0)*0.3
            originalty_list.append(originalty)
            
            attention = metrics['eye_openness']*0.3 + (1 - metrics['facial_symmetry'])*0.3 + au_values.get('AU04', 0)*0.4
            attention_list.append(attention)
        except Exception as e:
            print(f"Error processing result {i}: {str(e)}")
            # Continue with next result instead of failing completely
            continue
    
    if len(ideation_list) == 0:
        return {"error": "No valid data could be processed."}
    
    try:
        # Find the top image for each metric
        select = np.argmax([ideation_list, openness_list, originalty_list, attention_list], axis=1)    
        
        return {
            'ideation': {"balance": f"{np.mean(ideation_list)*100:.1f}%", "top_image": int(select[0])},
            'openness': {"balance": f"{np.mean(openness_list)*100:.1f}%", "top_image": int(select[1])},
            'originalty': {"balance": f"{np.mean(originalty_list)*100:.1f}%", "top_image": int(select[2])},
            'attention': {"balance": f"{np.mean(attention_list)*100:.1f}%", "top_image": int(select[3])},
        }
    except Exception as e:
        return {"error": f"Error calculating section 3: {str(e)}"}
