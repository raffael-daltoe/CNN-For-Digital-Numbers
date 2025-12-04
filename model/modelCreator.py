import os
import glob
import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw, ImageFilter
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt

# --- Configuration ---
FONT_FOLDER = '/tmp/ttf_collection'
MODEL_FILENAME = 'lcd_digit_model.h5'
IMAGE_SIZE = (32, 32)
SAMPLES_PER_DIGIT_PER_FONT = 500 
EPOCHS = 50
BATCH_SIZE = 32

def add_advanced_lcd_realism(image, digit):
    """Add advanced realistic LCD effects with digit-specific adjustments"""
    img = image.astype(np.float32)
    
    # Digit-specific segment simulation (different erosion for different digits)
    kernel_sizes = {
        0: (2, 2), 1: (1, 3), 2: (2, 2), 3: (2, 2), 4: (2, 2),
        5: (2, 2), 6: (2, 2), 7: (1, 2), 8: (2, 2), 9: (2, 2)
    }
    
    kernel_size = kernel_sizes.get(digit, (2, 2))
    kernel = np.ones(kernel_size, np.uint8)
    
    # Create segment gaps with morphological operations
    eroded = cv2.erode(image, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    
    # Add Gaussian blur with varying intensity
    blur_intensity = np.random.choice([1, 3, 5])
    blurred = cv2.GaussianBlur(dilated, (blur_intensity, blur_intensity), 0)
    
    # Add different types of noise
    noise_type = np.random.choice(['gaussian', 'salt_pepper', 'speckle'])
    if noise_type == 'gaussian':
        noise = np.random.normal(0, 8, blurred.shape)
        noisy = blurred + noise
    elif noise_type == 'salt_pepper':
        noisy = blurred.copy()
        salt_pepper_ratio = 0.01
        # Salt
        salt_coords = [np.random.randint(0, i-1, int(blurred.size * salt_pepper_ratio)) for i in blurred.shape]
        noisy[salt_coords[0], salt_coords[1]] = 255
        # Pepper
        pepper_coords = [np.random.randint(0, i-1, int(blurred.size * salt_pepper_ratio)) for i in blurred.shape]
        noisy[pepper_coords[0], pepper_coords[1]] = 0
    else:  # speckle
        noise = np.random.normal(0, 0.1, blurred.shape)
        noisy = blurred + blurred * noise
    
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    # Advanced brightness/contrast variations
    alpha = np.random.uniform(0.7, 1.4)  # Wider contrast range
    beta = np.random.uniform(-20, 20)    # Wider brightness range
    adjusted = cv2.convertScaleAbs(noisy, alpha=alpha, beta=beta)
    
    # Add perspective transformation occasionally
    if np.random.random() < 0.3:
        h, w = adjusted.shape
        pts1 = np.float32([[0,0], [w,0], [0,h], [w,h]])
        pts2 = np.float32([
            [np.random.randint(-5, 5), np.random.randint(-5, 5)],
            [w - np.random.randint(-5, 5), np.random.randint(-5, 5)],
            [np.random.randint(-5, 5), h - np.random.randint(-5, 5)],
            [w - np.random.randint(-5, 5), h - np.random.randint(-5, 5)]
        ])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        adjusted = cv2.warpPerspective(adjusted, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # Add motion blur occasionally
    if np.random.random() < 0.2:
        size = np.random.randint(3, 8)
        kernel_motion = np.zeros((size, size))
        kernel_motion[int((size-1)/2), :] = np.ones(size)
        kernel_motion /= size
        adjusted = cv2.filter2D(adjusted, -1, kernel_motion)
    
    return adjusted

def generate_digit_image(digit, font):
    """Generate digit image with proper centering and variations"""
    # Vary image size slightly
    width = np.random.randint(35, 45)
    height = np.random.randint(55, 65)
    
    image = Image.new('L', (width, height), 'black')
    draw = ImageDraw.Draw(image)
    
    # Vary font size slightly
    font_size = np.random.randint(42, 50)
    try:
        dynamic_font = font.font_variant(size=font_size)
    except:
        dynamic_font = font
    
    # Center the text with slight random offsets
    x_offset = np.random.randint(-2, 3)
    y_offset = np.random.randint(-2, 3)
    
    draw.text((width//2 + x_offset, height//2 + y_offset), str(digit), 
              fill='white', font=dynamic_font, anchor='mm')
    
    return np.array(image)

def create_enhanced_synthetic_dataset():
    """Creates an enhanced dataset with more realistic variations"""
    font_paths = glob.glob(os.path.join(FONT_FOLDER, '*.ttf')) + \
                 glob.glob(os.path.join(FONT_FOLDER, '*.TTF'))

    if not font_paths:
        print(f"Error: No font files found in '{FONT_FOLDER}' directory.")
        exit()

    print(f"Found {len(font_paths)} fonts. Generating enhanced synthetic dataset...")

    images = []
    labels = []

    # Enhanced data augmentation
    datagen = ImageDataGenerator(
        rotation_range=8,  # Increased rotation
        width_shift_range=0.15,  # Increased shift
        height_shift_range=0.15,
        shear_range=0.15,  # Increased shear
        zoom_range=[0.85, 1.15],  # Wider zoom range
        brightness_range=[0.7, 1.3],  # Wider brightness
        fill_mode='constant',
        cval=0
    )

    for font_path in font_paths:
        print(f"Processing font: {os.path.basename(font_path)}")
        
        try:
            font = ImageFont.truetype(font_path, 48)
        except:
            print(f"  Skipping font (load error): {font_path}")
            continue

        for digit in range(10):
            # Generate multiple base images with variations
            for _ in range(SAMPLES_PER_DIGIT_PER_FONT // 2):
                base_image = generate_digit_image(digit, font)
                
                # Add advanced realism
                realistic_image = add_advanced_lcd_realism(base_image, digit)
                
                # Reshape for data generator
                realistic_image = realistic_image.reshape((1,) + realistic_image.shape + (1,))

                count = 0
                for batch in datagen.flow(realistic_image, batch_size=1):
                    resized_img = cv2.resize(batch[0], IMAGE_SIZE)
                    images.append(resized_img)
                    labels.append(digit)
                    count += 1
                    if count >= 2:  # Generate 2 augmented versions per base image
                        break

    print(f"\nDataset generation complete. Total samples: {len(images)}")
    return np.array(images), np.array(labels)

def build_enhanced_model(input_shape, num_classes):
    """Builds an enhanced CNN model with better architecture"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    """Main function to generate data, build, train, and save the model."""
    X, y = create_enhanced_synthetic_dataset()

    X = X.reshape(X.shape[0], IMAGE_SIZE[0], IMAGE_SIZE[1], 1).astype('float32') / 255.0
    y = to_categorical(y, num_classes=10)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    model = build_enhanced_model(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1), num_classes=10)
    print("\n--- Model Summary ---")
    model.summary()

    # Enhanced callbacks
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    ]

    print("\n--- Starting Model Training ---")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy*100:.2f}%")

    model.save(MODEL_FILENAME)
    print(f"Model saved successfully as '{MODEL_FILENAME}'")

if __name__ == '__main__':
    main()