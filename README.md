# LCD Digit Recognition Pipeline

This project allows you to recognize numbers displayed on LCD screens recorded in video files. It uses a two-step process: first, training a deep learning model on synthetically generated realistic LCD digits, and second, applying that model to process a video frame-by-frame.

## Prerequisites

You need Python installed (preferably 3.8 - 3.10) along with the following libraries:

  * TensorFlow (and Keras)
  * OpenCV (`opencv-python`)
  * NumPy
  * Matplotlib
  * Pillow (PIL)
  * scikit-learn

You can install the necessary dependencies using the `requirements.txt` file:

```bash
pip install -r scripts/requirements.txt
```

-----

## Step 1: Training the Model (`modelCreator.py`)

Before analyzing videos, you must train the recognition model, or if you preferer, you can use the lcd_digit_model which is already commited here. This script generates synthetic images that look like LCD displays by applying various noise, blur, erosion, and perspective transformations to standard TrueType Fonts (TTF).

### Running Training

Run the script:

```bash
python model/modelCreator.py
```

-----

## Step 2: Analyzing Video (`readImage/read.py`)

Once you have the `lcd_digit_model.h5` file, you can use the video processing script to read digits from a video.

### Preparation

1.  **Save the Script:** Save the first code block provided as `read.py`.
2.  **Place Model:** Ensure the `lcd_digit_model.h5` file (generated in Step 1) is in the same directory where you call this script(`read.py`).
3.  **Prepare Video:** Have your target video file ready (e.g., `my_meter_reading.mp4`).

### Usage

Run the script from the command line. You need to specify three arguments:

1.  `--video`: Path to your video file.
2.  `--digits`: The total number of digits you want to read.
3.  `--decimal`: The position of the decimal point from the left (e.g., if the number is `12.34`, the position is 2). Use `0` if there is no decimal point.

#### Examples

**Scenario 1: Reading a 5-digit integer (e.g., "12345")**

```bash
python readImage/read.py --video inputs/meter.mp4 --digits 5 --decimal 0
```

**Scenario 2: Reading a number with 2 digits before and 2 after decimal (e.g., "12.34")**
Total digits is 4. Decimal position is 2.

```bash
python readImage/read.py --video inputs/pressure_gauge.avi --digits 4 --decimal 2
```

### Interactive Selection Process

When the script starts, it will pause on the first frame of the video and open a window named "Select Digits". You must manually define the region for **each** digit sequentially.

**Selection Controls:**

  * **Left Mouse Click:** Click 4 corners to define the bounding box for the *current* digit being asked for (e.g., "Select Digit 1 of 4"). Order of corners does not matter.
  * **`c` key (Confirm):** Once you have selected 4 points and a green box appears around the digit, press 'c' to confirm. The box will turn red and become permanent. The interface will move to the next digit.
  * **`r` key (Reset):** If you misplaced a point for the *current* digit, press 'r' to clear points and try again.
  * **`ESC` key:** Quit the application.

**Procedure:**

1.  Select the 4 corners of the first digit (leftmost). Press `c`.
2.  Select the 4 corners of the second digit. Press `c`.
3.  Repeat until all digits specified in the `--digits` argument are selected.

### Output

Once selections are complete, the video will start playing.

1.  A live view will show recognized values overlaid on the video.
2.  The results will be saved to **`video_analysis_results.csv`**.

**CSV Format:**
The CSV will contain the timestamp (second), the combined full value (including the decimal if configured), and the individual raw readings for each digit region.

```csv
Second,Full Value,Digit_1,Digit_2,Digit_3,Digit_4
0,12.34,1,2,3,4
1,12.35,1,2,3,5
```