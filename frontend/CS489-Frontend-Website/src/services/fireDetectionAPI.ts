// const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5174';

// export interface PredictResponse {
//   predicted_class: 'fire' | 'no_fire';
//   probability_fire: number;
//   probability_no_fire: number;
// }

// export interface BoundingBox {
//   x1: number;
//   y1: number;
//   x2: number;
//   y2: number;
// }

// export interface DetectedObject {
//   name: string;
//   class: number;
//   confidence: number;
//   box: BoundingBox;
// }

// export interface DetectResponse {
//   image: string; // base64 encoded image with bounding boxes
//   predicted_boxes: DetectedObject[];
// }

// export class FireDetectionAPI {
// //  Classify image as fire or no fire
//   static async predict(imageFile: File): Promise<PredictResponse> {
//     const formData = new FormData();
//     formData.append('image', imageFile);

//     const response = await fetch(`${API_BASE_URL}/predict`, {
//       method: 'POST',
//       body: formData,
//     });

//     if (!response.ok) {
//       const error = await response.json();
//       throw new Error(error.detail || 'Failed to classify image');
//     }

//     return response.json();
//   }
// // Detect fire with YOLO and get bounding boxes
//   static async detect(imageFile: File): Promise<DetectResponse> {
//     const formData = new FormData();
//     formData.append('image', imageFile);

//     const response = await fetch(`${API_BASE_URL}/detect`, {
//       method: 'POST',
//       body: formData,
//     });

//     if (!response.ok) {
//       const error = await response.json();
//       throw new Error(error.detail || 'Failed to detect objects');
//     }

//     return response.json();
//   }
// }

// src/services/fireDetectionApi.ts

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface PredictResponse {
  predicted_class: 'fire' | 'no_fire';
  probability_fire: number;
  probability_no_fire: number;
}

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface DetectedObject {
  name: string;
  class: number;
  confidence: number;
  box: BoundingBox;
}

export interface DetectResponse {
  image: string;
  predicted_boxes: DetectedObject[];
}

export class FireDetectionAPI {
  static async predict(imageFile: File): Promise<PredictResponse> {
    const formData = new FormData();
    formData.append('image', imageFile);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      // Log the response for debugging
      console.log('Response status:', response.status);
      console.log('Response headers:', response.headers);

      if (!response.ok) {
        // Try to get error details
        const text = await response.text();
        console.error('Error response:', text);
        
        try {
          const error = JSON.parse(text);
          throw new Error(error.detail || `API Error: ${response.status}`);
        } catch {
          throw new Error(`API Error: ${response.status} - ${text || 'Unknown error'}`);
        }
      }

      const text = await response.text();
      console.log('Response body:', text);
      
      // Check if response is empty
      if (!text) {
        throw new Error('Empty response from server');
      }

      return JSON.parse(text);
    } catch (error) {
      if (error instanceof Error) {
        throw error;
      }
      throw new Error('Failed to connect to API. Is the backend running?');
    }
  }

  static async detect(imageFile: File): Promise<DetectResponse> {
    const formData = new FormData();
    formData.append('image', imageFile);

    try {
      const response = await fetch(`${API_BASE_URL}/detect`, {
        method: 'POST',
        body: formData,
      });

      console.log('Detect response status:', response.status);

      if (!response.ok) {
        const text = await response.text();
        console.error('Error response:', text);
        
        try {
          const error = JSON.parse(text);
          throw new Error(error.detail || `API Error: ${response.status}`);
        } catch {
          throw new Error(`API Error: ${response.status} - ${text || 'Unknown error'}`);
        }
      }

      const text = await response.text();
      console.log('Detect response body:', text);
      
      if (!text) {
        throw new Error('Empty response from server');
      }

      return JSON.parse(text);
    } catch (error) {
      if (error instanceof Error) {
        throw error;
      }
      throw new Error('Failed to connect to API. Is the backend running?');
    }
  }
}
