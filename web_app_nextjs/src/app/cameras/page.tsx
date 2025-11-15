'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Camera, Plus, Settings, Wifi, WifiOff, Video, Cog } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useState } from 'react';

export default function CamerasPage() {
  const [hoveredCamera, setHoveredCamera] = useState<number | null>(null);
  const cameras = [
    { id: 1, name: 'Entrance Main', location: 'Building A - Floor 1', status: 'online', alerts: 3, videoUrl: '/videos/cameras/14698511_1920_1080_60fps.mp4' },
    { id: 2, name: 'Parking Lot', location: 'Outdoor - North', status: 'online', alerts: 0, videoUrl: '/videos/cameras/New York 1956 42nd St WebCam - LIVE.mp4' },
    { id: 3, name: 'Hallway B2', location: 'Building B - Floor 2', status: 'online', alerts: 1, videoUrl: '/videos/cameras/5108891-uhd_3840_2160_30fps.mp4' },
    { id: 4, name: 'Cafeteria', location: 'Building A - Floor 1', status: 'offline', alerts: 0, videoUrl: null },
    { id: 5, name: 'Stairwell A', location: 'Building A - Floor 3', status: 'online', alerts: 0, videoUrl: '/videos/cameras/2697636-uhd_1920_1440_30fps.mp4' },
    { id: 6, name: 'Lobby', location: 'Main Building', status: 'online', alerts: 2, videoUrl: '/videos/cameras/5977704-hd_1366_586_30fps.mp4' },
  ];

  return (
    <div className="p-4 sm:p-6 md:p-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-6 sm:mb-8">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold text-[var(--text-primary)] mb-2">
            Camera Management
          </h1>
          <p className="text-sm sm:text-base text-[var(--text-secondary)]">
            Monitor and configure your security cameras
          </p>
        </div>
        <Button className="w-full sm:w-auto gap-2 bg-[var(--accent-blue)] hover:bg-blue-600">
          <Plus className="h-4 w-4" />
          Add Camera
        </Button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 sm:gap-6 mb-6 sm:mb-8">
        <Card className="border-[var(--border)] bg-[var(--card-bg)]">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-[var(--text-secondary)]">
              Total Cameras
            </CardTitle>
            <Camera className="h-4 w-4 text-blue-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-[var(--text-primary)]">24</div>
            <p className="text-xs text-[var(--text-secondary)] mt-1">+2 this month</p>
          </CardContent>
        </Card>

        <Card className="border-[var(--border)] bg-[var(--card-bg)]">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-[var(--text-secondary)]">
              Online
            </CardTitle>
            <Wifi className="h-4 w-4 text-green-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-[var(--text-primary)]">23</div>
            <p className="text-xs text-green-500 mt-1">95.8% uptime</p>
          </CardContent>
        </Card>

        <Card className="border-[var(--border)] bg-[var(--card-bg)]">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-[var(--text-secondary)]">
              Active Alerts
            </CardTitle>
            <WifiOff className="h-4 w-4 text-orange-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-[var(--text-primary)]">6</div>
            <p className="text-xs text-[var(--text-secondary)] mt-1">Across all cameras</p>
          </CardContent>
        </Card>
      </div>

      {/* Camera Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
        {cameras.map((camera) => (
          <Card key={camera.id} className="border-[var(--border)] bg-[var(--card-bg)]">
            <CardHeader className="flex flex-row items-center justify-between pb-3">
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${
                  camera.status === 'online' ? 'bg-green-500' : 'bg-red-500'
                }`} />
                <CardTitle className="text-base text-[var(--text-primary)]">
                  {camera.name}
                </CardTitle>
              </div>
              <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                <Settings className="h-4 w-4" />
              </Button>
            </CardHeader>
            <CardContent>
              {/* Camera Preview */}
              <div
                className="aspect-video bg-black rounded-lg mb-3 relative overflow-hidden group cursor-pointer"
                onMouseEnter={() => setHoveredCamera(camera.id)}
                onMouseLeave={() => setHoveredCamera(null)}
              >
                {camera.status === 'online' && camera.videoUrl ? (
                  <>
                    <video
                      key={camera.id}
                      className="w-full h-full object-cover"
                      loop
                      muted
                      playsInline
                      preload="metadata"
                      ref={(video) => {
                        if (video) {
                          if (hoveredCamera === camera.id) {
                            video.play().catch(() => {});
                          } else {
                            video.pause();
                            video.currentTime = 0;
                          }
                        }
                      }}
                    >
                      <source src={camera.videoUrl} type="video/mp4" />
                      Your browser does not support the video tag.
                    </video>
                    <div className="absolute top-2 left-2 bg-black/70 px-2 py-1 rounded-full flex items-center gap-1.5">
                      <div className={`w-1.5 h-1.5 rounded-full ${hoveredCamera === camera.id ? 'bg-red-500 animate-pulse' : 'bg-gray-500'}`} />
                      <span className="text-xs text-white font-medium">{hoveredCamera === camera.id ? 'LIVE' : 'PAUSED'}</span>
                    </div>
                    {hoveredCamera !== camera.id && (
                      <div className="absolute inset-0 flex items-center justify-center bg-black/40 group-hover:bg-black/20 transition-colors">
                        <div className="bg-white/20 backdrop-blur-sm rounded-full p-4 group-hover:scale-110 transition-transform">
                          <Video className="h-8 w-8 text-white" />
                        </div>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="w-full h-full flex items-center justify-center bg-[var(--border)]">
                    <Camera className="h-12 w-12 text-[var(--text-secondary)] opacity-50" />
                  </div>
                )}
              </div>

              {/* Camera Info */}
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-[var(--text-secondary)]">Location:</span>
                  <span className="text-[var(--text-primary)] font-medium">{camera.location}</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-[var(--text-secondary)]">Status:</span>
                  <span className={`font-medium ${
                    camera.status === 'online' ? 'text-green-500' : 'text-red-500'
                  }`}>
                    {camera.status === 'online' ? 'Online' : 'Offline'}
                  </span>
                </div>
                {camera.alerts > 0 && (
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-[var(--text-secondary)]">Alerts:</span>
                    <span className="text-orange-500 font-medium">{camera.alerts} active</span>
                  </div>
                )}
              </div>

              {/* Actions */}
              <div className="flex gap-2 mt-4">
                <Button
                  size="sm"
                  className="flex-1 bg-red-600 hover:bg-red-700 text-white font-semibold shadow-md shadow-red-500/30 border-0"
                >
                  <Video className="h-4 w-4 mr-1.5" />
                  View Live
                </Button>
                <Button
                  size="sm"
                  className="flex-1 bg-[#1a2942] hover:bg-[#0a1929] text-[var(--text-primary)] font-semibold border border-[var(--accent-blue)] shadow-md shadow-blue-500/20"
                >
                  <Cog className="h-4 w-4 mr-1.5" />
                  Configure
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
