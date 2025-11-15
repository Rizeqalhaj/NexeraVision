import type { UploadResponse, DetectionResult } from '@/types/detection';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001/api';

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

export async function uploadVideo(file: File): Promise<DetectionResult> {
  const formData = new FormData();
  formData.append('video', file);

  try {
    const response = await fetch(`${API_BASE_URL}/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new ApiError(response.status, `Upload failed: ${response.statusText}`);
    }

    const data: UploadResponse = await response.json();

    if (!data.success || !data.data) {
      throw new Error(data.error || 'Upload failed');
    }

    return data.data;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    throw new Error(`Network error: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

export async function uploadWithProgress(
  file: File,
  onProgress: (progress: number) => void
): Promise<DetectionResult> {
  const formData = new FormData();
  formData.append('video', file);

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();

    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable) {
        const progress = Math.round((e.loaded / e.total) * 100);
        onProgress(progress);
      }
    });

    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const response: UploadResponse = JSON.parse(xhr.responseText);
          if (response.success && response.data) {
            resolve(response.data);
          } else {
            reject(new Error(response.error || 'Upload failed'));
          }
        } catch (error) {
          reject(new Error('Failed to parse response'));
        }
      } else {
        reject(new ApiError(xhr.status, `Upload failed: ${xhr.statusText}`));
      }
    });

    xhr.addEventListener('error', () => {
      reject(new Error('Network error occurred'));
    });

    xhr.open('POST', `${API_BASE_URL}/upload`);
    xhr.send(formData);
  });
}

/**
 * Model Abstraction Layer
 * Supports multiple AI models for A/B testing and gradual migration
 */
export interface DetectionModel {
  id: string;
  name: string;
  version: string;
  endpoint: string;
  type: 'legacy' | 'modern' | 'experimental';
  preprocess?: (frames: string[]) => string[];
  postprocess?: (results: DetectionResult[]) => DetectionResult[];
}

/**
 * Available detection models
 * Add new models here without changing calling code
 */
export const DETECTION_MODELS: Record<string, DetectionModel> = {
  'vgg19-legacy': {
    id: 'vgg19-legacy',
    name: 'VGG19 + Bi-LSTM',
    version: '1.0',
    endpoint: '/api/detect/video',
    type: 'legacy',
  },
  'modern-model': {
    id: 'modern-model',
    name: 'Modern Architecture',
    version: '2.0',
    endpoint: '/api/v2/detect',
    type: 'modern',
  },
  'experimental': {
    id: 'experimental',
    name: 'Experimental Model',
    version: '3.0-beta',
    endpoint: '/api/v3/detect',
    type: 'experimental',
  },
};

/**
 * Get active model (can be configured via env or user preference)
 */
export function getActiveModel(): DetectionModel {
  const modelId = process.env.NEXT_PUBLIC_ACTIVE_MODEL || 'vgg19-legacy';
  return DETECTION_MODELS[modelId] || DETECTION_MODELS['vgg19-legacy'];
}

export function createWebSocketConnection(url?: string): WebSocket {
  const wsUrl = url || process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:3001/ws/live';
  return new WebSocket(wsUrl);
}

export async function detectViolenceBatch(
  imageDataArray: string[]
): Promise<{ violenceProbability: number }[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/detect/batch`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        images: imageDataArray,
      }),
    });

    if (!response.ok) {
      throw new ApiError(response.status, `Detection failed: ${response.statusText}`);
    }

    const data = await response.json();

    if (!data.success || !data.results) {
      throw new Error(data.error || 'Detection failed');
    }

    return data.results;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    throw new Error(`Network error: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

export async function detectViolenceSingle(
  imageData: string
): Promise<{ violenceProbability: number }> {
  try {
    const response = await fetch(`${API_BASE_URL}/detect/image`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image: imageData,
      }),
    });

    if (!response.ok) {
      throw new ApiError(response.status, `Detection failed: ${response.statusText}`);
    }

    const data = await response.json();

    if (!data.success || !data.result) {
      throw new Error(data.error || 'Detection failed');
    }

    return data.result;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    throw new Error(`Network error: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}
