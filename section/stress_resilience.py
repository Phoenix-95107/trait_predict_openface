import numpy as np
import pandas as pd
import math

class StressResilienceAnalyzer:
    def __init__(self):
        # No need for MediaPipe initialization anymore
        pass
    
    def process_openface_data(self, openface_data):
        """
        Process OpenFace data and return forehead furrows and lip compression metrics.
        Args:
            openface_data: DataFrame containing OpenFace analysis results
        Returns:
            Dictionary containing forehead furrows and lip compression scores
        """
        try:
            if openface_data.empty:
                return {"error": "No data provided"}
            
            # Calculate metrics using OpenFace data
            forehead_furrows = self.calculate_forehead_furrows_openface(openface_data)
            lip_compression = self.calculate_lip_compression_openface(openface_data)
            
            return {
                "forehead_furrows": forehead_furrows,
                "lip_compression": lip_compression
            }
        except Exception as e:
            return {"error": str(e)}
    
    def calculate_forehead_furrows_openface(self, openface_data):
        """
        Calculate forehead furrows based on OpenFace data.
        Args:
            openface_data: DataFrame containing OpenFace analysis results
        Returns:
            Forehead furrows score (0-1, where 1 is high furrow presence)
        """
        try:

            forehead_aus = [' AU01_r', ' AU02_r', ' AU04_r']
            au_values = []
            
            for au in forehead_aus:
                if au in openface_data.keys():
                    au_values.append(openface_data[au])
            
            if not au_values:
                return 0.5  # Default value if no relevant AUs found
            
            # Calculate forehead furrows score
            # Higher AU04 (brow lowerer) indicates more furrows
            # AU01 and AU02 can also contribute to forehead lines when active
            
            # Normalize AU values (OpenFace typically uses 0-5 scale)
            normalized_aus = [min(val / 5.0, 1.0) for val in au_values]
            
            # Weight AU04 more heavily for furrows
            if len(normalized_aus) == 3:  # If we have all three AUs
                # AU04 is most important for furrows, followed by AU01 and AU02
                weighted_score = normalized_aus[2] * 0.6 + normalized_aus[0] * 0.2 + normalized_aus[1] * 0.2
            else:
                # Simple average if we don't have all AUs
                weighted_score = sum(normalized_aus) / len(normalized_aus)
            
            return weighted_score
            
        except Exception as e:
            print(f"Error calculating forehead furrows: {e}")
            return 0.5  # Default value on error
    
    def calculate_lip_compression_openface(self, openface_data):
        """
        Calculate lip compression based on OpenFace data.
        
        Args:
            openface_data: DataFrame containing OpenFace analysis results
            
        Returns:
            Lip compression score (0-1, where 1 is high compression)
        """
        try:
            # Extract relevant AUs for lip compression
            lip_aus = [' AU23_r', ' AU24_r', ' AU28_r']
            au_values = []
            
            for au in lip_aus:
                if au in openface_data.keys():
                    au_values.append(openface_data[au])
            
            # If we don't have the specific lip compression AUs, 
            # we can use other mouth-related AUs as a fallback
            if not au_values:
                fallback_aus = ['AU14_r', 'AU15_r', 'AU17_r', 'AU20_r']
                for au in fallback_aus:
                    if au in openface_data.keys():
                        au_values.append(openface_data[au])
            
            if not au_values:
                return 0.5  # Default value if no relevant AUs found
            
            # Normalize AU values (OpenFace typically uses 0-5 scale)
            normalized_aus = [min(val / 5.0, 1.0) for val in au_values]
            
            # Calculate lip compression score
            lip_compression = sum(normalized_aus) / len(normalized_aus)
            
            return lip_compression
            
        except Exception as e:
            print(f"Error calculating lip compression: {e}")
            return 0.5  # Default value on error

def calculate_section4(openface_results):
    analyzer = StressResilienceAnalyzer()
    # stress_indicator = []
    emotional_regulation = []
    resilience_score = []
    
    # Check if we have valid data
    if not openface_results or isinstance(openface_results, dict) and 'error' in openface_results:
        return {"error": "No valid OpenFace data provided."}
    
    # Process each result (each result corresponds to one image/frame)
    df = pd.read_csv(openface_results)
    for i, result_data in df.iterrows():
        try:
            # Convert the result to a DataFrame if it's not already
            # Process the OpenFace data
            metrics = analyzer.process_openface_data(result_data)
            
            if 'error' in metrics:
                print(f"Error processing data {i}: {metrics['error']}")
                continue
            
            # Extract Action Units (AUs) from OpenFace data
            au_values = {}
            for au in ['AU04', 'AU12', 'AU15', 'AU24']:
                au_col = f" {au}_r"  # OpenFace uses _r suffix for AU intensity
                if au_col in result_data.keys():
                    au_values[au] = result_data[au_col].mean() / 5.0  # Normalize to 0-1 range
                else:
                    au_values[au] = 0.0  # Default if AU not available
            
            # Calculate stress resilience metrics using the same formulas
            # stress = au_values.get('AU04', 0)*0.4 + au_values.get('AU24', 0)*0.3 + au_values.get('AU04', 0)*0.3
            # stress_indicator.append(stress)
           
            emotional = (1 - metrics['forehead_furrows'])*0.5 + (1 - au_values.get('AU04', 0))*0.3 + (1 - au_values.get('AU15', 0))*0.2
            emotional_regulation.append(emotional)
            
            resilience = (1-stress)*0.4 + emotional*0.4 + au_values.get('AU12', 0)*0.2
            resilience_score.append(resilience)
            
        except Exception as e:
            print(f"Error processing result {i}: {str(e)}")
            # Continue with next result instead of failing completely
            continue
    
    # Only proceed if we have processed at least one image successfully
    try:
        select = np.argmax([emotional_regulation, resilience_score], axis=1)
        return {
            # 'stress_indicator': {"balance": f"{np.mean(stress_indicator)*100:.1f}%", "top_image": int(select[0])},
            'emotional_regulation': {"balance": f"{np.mean(emotional_regulation)*100:.1f}%", "top_image": int(select[0])},
            'resilience_score': {"balance": f"{np.mean(resilience_score)*100:.1f}%", "top_image": int(select[1])},
        }
    except Exception as e:
        return {"error": f"Failed to calculate final metrics: {str(e)}"}
