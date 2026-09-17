import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import pickle
import os

# Model file paths
MODEL_PATH = 'digit_model.pkl'
SCALER_PATH = 'scaler.pkl'

class DigitRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Handwritten Digit Recognition')
        self.root.geometry('600x750')
        self.root.resizable(False, False)

        # Load or create model
        self.model = self.load_or_create_model()

        # Canvas settings
        self.canvas_size = 280
        self.brush_size = 18
        self.drawing = False

        # Create in-memory image
        self.image = Image.new('L', (280, 280), color=255)
        self.draw = ImageDraw.Draw(self.image)

        # GUI Components
        self.setup_ui()

    def load_or_create_model(self):
        """Load existing model or create and train a new one"""
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            print("Loading pre-trained model...")
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            with open(SCALER_PATH, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            print("Training a new high-accuracy model...")
            model = self.train_and_save_model()
        return model

    def augment_data(self, images):
        """Augment dataset with rotations and shifts for better generalization"""
        augmented = [images]

        print("Augmenting training data...")
        for angle in [-15, -10, -5, 5, 10, 15]:
            rotated = []
            for img in images:
                img_8x8 = img.reshape(8, 8)
                img_pil = Image.fromarray((img_8x8 * 255).astype('uint8'))
                img_rot = img_pil.rotate(angle, expand=False, fillcolor=255)
                img_array = np.array(img_rot, dtype='float32') / 255.0
                rotated.append(img_array.flatten())
            augmented.append(np.array(rotated))

        return np.vstack(augmented)

    def train_and_save_model(self):
        """Train XGBoost model on augmented digits data"""

        # Load sklearn digits dataset
        print("Loading sklearn digits dataset...")
        digits = load_digits()
        x_data = digits.data / 16.0  # Normalize
        y_data = digits.target

        # Augment data for better accuracy
        x_augmented = self.augment_data(x_data)
        y_augmented = np.tile(y_data, len(x_augmented) // len(y_data))

        print(f"Dataset size after augmentation: {x_augmented.shape}")

        # Split data
        x_train, x_test, y_train, y_test = train_test_split(
            x_augmented, y_augmented, test_size=0.2, random_state=42
        )

        # Scale features
        print("Scaling features...")
        self.scaler = StandardScaler()
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)

        # Train XGBoost model
        print("Training XGBoost model...")
        try:
            import xgboost as xgb
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                verbose=1,
                random_state=42,
                tree_method='hist'
            )
        except ImportError:
            print("XGBoost not available, using Gradient Boosting...")
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                subsample=0.9,
                verbose=1,
                random_state=42
            )

        model.fit(x_train_scaled, y_train)

        # Evaluate
        test_accuracy = model.score(x_test_scaled, y_test)
        print(f"\n{'='*50}")
        print(f"Test accuracy: {test_accuracy*100:.2f}%")
        print(f"{'='*50}\n")

        # Save
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(self.scaler, f)

        return model

    def setup_ui(self):
        """Create the user interface"""
        title_label = tk.Label(
            self.root,
            text='Handwritten Digit Recognition',
            font=('Arial', 18, 'bold')
        )
        title_label.pack(pady=10)

        instructions = tk.Label(
            self.root,
            text='Draw a digit (0-9) on the canvas using your mouse',
            font=('Arial', 10)
        )
        instructions.pack(pady=5)

        self.canvas = tk.Canvas(
            self.root,
            width=self.canvas_size,
            height=self.canvas_size,
            bg='white',
            cursor='cross',
            relief=tk.SUNKEN,
            bd=2
        )
        self.canvas.pack(pady=20)

        self.canvas.bind('<Button-1>', self.on_mouse_press)
        self.canvas.bind('<B1-Motion>', self.on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_release)

        result_frame = tk.Frame(self.root)
        result_frame.pack(pady=10)

        tk.Label(result_frame, text='Digit:', font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
        self.digit_label = tk.Label(
            result_frame,
            text='?',
            font=('Arial', 20, 'bold'),
            fg='blue',
            width=3
        )
        self.digit_label.pack(side=tk.LEFT, padx=5)

        tk.Label(result_frame, text='Confidence:', font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
        self.confidence_label = tk.Label(
            result_frame,
            text='0%',
            font=('Arial', 16, 'bold'),
            fg='green'
        )
        self.confidence_label.pack(side=tk.LEFT, padx=5)

        clear_button = tk.Button(
            self.root,
            text='Clear Canvas',
            command=self.clear_canvas,
            font=('Arial', 12),
            width=20,
            bg='#FF6B6B',
            fg='white'
        )
        clear_button.pack(pady=10)

        info_label = tk.Label(
            self.root,
            text='Release the mouse button to predict',
            font=('Arial', 9),
            fg='gray'
        )
        info_label.pack(pady=5)

    def on_mouse_press(self, event):
        self.drawing = True

    def on_mouse_drag(self, event):
        if not self.drawing:
            return

        x1 = event.x - self.brush_size // 2
        y1 = event.y - self.brush_size // 2
        x2 = event.x + self.brush_size // 2
        y2 = event.y + self.brush_size // 2

        self.canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')
        self.draw.ellipse([(x1, y1), (x2, y2)], fill=0)

    def on_mouse_release(self, event):
        self.drawing = False
        self.predict_digit()

    def predict_digit(self):
        """Preprocess and predict"""
        img_array = np.array(self.image, dtype='float32')
        img_array = 255 - img_array  # Invert
        img_array = img_array / 255.0  # Normalize

        # Resize to 8x8 (to match training data format)
        img_resized = Image.fromarray((img_array * 255).astype('uint8'))
        img_resized = img_resized.resize((8, 8), Image.Resampling.LANCZOS)
        img_resized = np.array(img_resized, dtype='float32') / 255.0

        img_input = img_resized.flatten().reshape(1, -1)
        img_scaled = self.scaler.transform(img_input)

        predictions = self.model.predict_proba(img_scaled)
        predicted_digit = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_digit]) * 100

        self.digit_label.config(text=str(predicted_digit))
        self.confidence_label.config(text=f'{confidence:.1f}%')

    def clear_canvas(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (280, 280), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.digit_label.config(text='?')
        self.confidence_label.config(text='0%')

def main():
    root = tk.Tk()
    app = DigitRecognitionApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
