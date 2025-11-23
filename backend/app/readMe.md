

-----

## Fire Detection API Documentation

This backend provides two endpoints for **image-based fire classification** and **object detection**.

-----

### 1\. `POST /predict`

**Description**

Classifies an uploaded image as either **fire** or **no\_fire** using a Convolutional Neural Network (CNN) model.

#### Request

  * **Method:** `POST`
  * **Content-Type:** `multipart/form-data`
  * **Body:**
      * **image:** File (required) - The image to be classified.

#### Successful Response — `202 Accepted`

```json
{
  "predicted_class": "fire" | "no_fire",
  "probability_fire": 0.985, 
  "probability_no_fire": 0.015
}
```

#### Return Types

| Variable | Type | Description |
| :--- | :--- | :--- |
| **predicted\_class** | `string` | The resulting classification. |
| **probability\_fire** | `number` (float) | The model's confidence that the image contains fire (0.0 to 1.0). |
| **probability\_no\_fire** | `number` (float) | The model's confidence that the image contains no fire (0.0 to 1.0). |

#### Error Response — `400 Bad Request`

```json
{ 
  "detail": "Uploaded file is not an image" 
}
```

-----

### 2\. `POST /detect`

**Description**

Runs **YOLO (You Only Look Once)** object detection on the uploaded image. It returns the annotated image and a summary of the detection output.

#### Request

  * **Method:** `POST`
  * **Content-Type:** `multipart/form-data`
  * **Body:**
      * **image:** File (required) - The image for object detection.

#### Successful Response — `200 OK`

```json
{
  "image": "base64-encoded JPEG string",
  "predicted_boxes": [
    {
      "name": "smoke",
      "class": 1,
      "confidence": 0.93033,
      "box": {
        "x1": 0.0,
        "y1": 0.0,
        "x2": 1109.73022,
        "y2": 611.08344
      }
    }
  ]
}
// Note: If no fire or smoke is detected, the "predicted_boxes" array will be empty:
// { "image": "...", "predicted_boxes": [] }
```

#### Return Types

| Variable | Type | Description |
| :--- | :--- | :--- |
| **image** | `string` | A **base64-encoded JPEG** string of the uploaded image with YOLO bounding boxes drawn on it. |
| **predicted\_boxes** | `array` (of objects) | A list of detected objects (fire/smoke). **This array will be empty (`[]`) if no objects are detected.** |

#### `predicted_boxes` Object Structure

| Key | Type | Description |
| :--- | :--- | :--- |
| **name** | `string` | The human-readable name of the detected object (e.g., "fire", "smoke"). |
| **class** | `number` (integer) | The numerical class ID of the detected object. |
| **confidence** | `number` (float) | The confidence score of the detection (0.0 to 1.0). |
| **box** | `object` | An object containing the bounding box coordinates. |
| **box.x1/y1** | `number` (float) | The normalized **top-left** coordinates of the bounding box. |
| **box.x2/y2** | `number` (float) | The normalized **bottom-right** coordinates of the bounding box. |

#### Error Response — `400 Bad Request`

```json
{ 
  "detail": "Uploaded file is not an image" 
}
```

-----

Let me know if you have any other adjustments or need help with code examples for these endpoints\!