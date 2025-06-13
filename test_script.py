import os
import warnings
import logging

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress other warnings
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from model_runner import ModelRunner

def test_model_on_training_data(model_path, training_folder):
    """Test how the model scores images from the training data"""
    
    print(f"Loading model from: {model_path}")
    print(f"Testing on images from: {training_folder}")
    print("-" * 50)
    
    # Check if paths exist
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return
        
    if not os.path.exists(training_folder):
        print(f"ERROR: Training folder not found: {training_folder}")
        return
    
    # Load model
    try:
        runner = ModelRunner(model_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return
    
    # Find all image files
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = []
    
    for filename in os.listdir(training_folder):
        if filename.lower().endswith(valid_extensions):
            image_files.append(filename)
    
    if not image_files:
        print(f"ERROR: No image files found in {training_folder}")
        print(f"Looking for files with extensions: {valid_extensions}")
        return
    
    print(f"Found {len(image_files)} images to test")
    print("-" * 50)
    
    # Test images
    scores = []
    predictions = []
    
    # Test up to 20 images
    for i, filename in enumerate(image_files[:20]):
        path = os.path.join(training_folder, filename)
        try:
            result = runner.score_image(path)

            # record for later statistics
            scores.append(result.get('anomaly_percentage'))
            predictions.append(result.get('prediction'))

            # pull individual fields, replacing None with a readable placeholder
            label         = result.get('label')                    # -1 / 1 / None
            label_str     = str(label) if label is not None else 'na'

            pred_txt      = result.get('prediction') or 'n/a'
            anomaly_pct   = result.get('anomaly_percentage')
            anomaly_str   = f"{anomaly_pct:5.1f}%" if anomaly_pct is not None else "  n/a"

            status_txt    = result.get('status', 'n/a')
            status_emoji  = "✅" if label == 1 else "❌"

            print(
                f"{status_emoji} {filename[:30]:30s} "
                f"| Anomaly: {anomaly_str:<6} "
                f"| Label: {label_str:>2} "
                f"| Pred: {pred_txt:<9} "
                f"| Status: {status_txt:<8}"
            )

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")


    
    # Summary statistics
    if scores:
        print("-" * 50)
        print(f"SUMMARY:")
        print(f"Images tested: {len(scores)}")
        print(f"Average anomaly level: {sum(scores)/len(scores):.1f}%")
        print(f"Min anomaly: {min(scores):.1f}%")
        print(f"Max anomaly: {max(scores):.1f}%")
        print(f"Normal images: {predictions.count(1)} ({predictions.count(1)/len(predictions)*100:.0f}%)")
        print(f"Anomalies detected: {predictions.count(-1)} ({predictions.count(-1)/len(predictions)*100:.0f}%)")
        
        # Warning if training images show high anomaly
        avg_anomaly = sum(scores)/len(scores)
        if avg_anomaly > 30:
            print("\n⚠️  WARNING: Training images showing high anomaly levels!")
            print("This suggests the model may need retraining with updated parameters.")
    else:
        print("ERROR: No images could be processed")

# Main execution
if __name__ == "__main__":
    # Update these paths to match your setup
    MODEL_PATH = r"\Models\defect_detector_20250612_145208.pkl"  # Update with your model filename
    TRAINING_FOLDER = r"\classifier\person1"  # Update with your training folder
    
    # You can also pass these as command line arguments
    import sys
    if len(sys.argv) >= 3:
        MODEL_PATH = sys.argv[1]
        TRAINING_FOLDER = sys.argv[2]
    
    test_model_on_training_data(MODEL_PATH, TRAINING_FOLDER)
