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

export function createWebSocketConnection(url?: string): WebSocket {
  const wsUrl = url || process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:3001/ws/live';
  return new WebSocket(wsUrl);
}
