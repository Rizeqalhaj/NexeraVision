'use client';

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Camera, MessageSquare, TrendingUp, CheckCircle2, Video, Play, X, BarChart3 } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function Dashboard() {
  const [showCameraGrid, setShowCameraGrid] = useState(false);
  const stats = [
    {
      title: 'Active Cameras',
      value: '5',
      subtitle: 'All',
      icon: Camera,
      iconColor: 'text-green-500',
      bgColor: 'bg-green-500/10',
    },
    {
      title: "Today's Incidents",
      value: '12',
      subtitle: '+3',
      icon: TrendingUp,
      iconColor: 'text-yellow-500',
      bgColor: 'bg-yellow-500/10',
    },
    {
      title: 'False Positive Rate',
      value: '4.2%',
      icon: CheckCircle2,
      iconColor: 'text-blue-500',
      bgColor: 'bg-blue-500/10',
    },
    {
      title: 'System Uptime',
      value: '99.8%',
      icon: CheckCircle2,
      iconColor: 'text-green-500',
      bgColor: 'bg-green-500/10',
    },
  ];

  const recentIncidents = [
    {
      id: 1,
      camera: 'Camera 12 - Parking Lot',
      time: '2 mins ago',
      type: 'Violence Detected',
      confidence: '94%',
      status: 'danger',
    },
    {
      id: 2,
      camera: 'Camera 45 - Entrance',
      time: '15 mins ago',
      type: 'Possible Incident',
      confidence: '78%',
      status: 'warning',
    },
  ];

  // Generate data for last 24 hours with peak at 18:00
  const generateChartData = () => {
    const data = [];
    const now = new Date();

    for (let i = 23; i >= 0; i--) {
      const hour = new Date(now.getTime() - i * 60 * 60 * 1000);
      const hourValue = hour.getHours();

      // Create peak at 18:00 (6 PM)
      let incidents;
      if (hourValue === 18) {
        incidents = 8; // Peak
      } else if (hourValue === 17 || hourValue === 19) {
        incidents = 5; // Around peak
      } else if (hourValue >= 9 && hourValue <= 21) {
        incidents = Math.floor(Math.random() * 3) + 2; // Daytime activity
      } else {
        incidents = Math.floor(Math.random() * 2); // Night time low activity
      }

      data.push({
        time: `${hourValue.toString().padStart(2, '0')}:00`,
        incidents: incidents,
      });
    }

    return data;
  };

  const chartData = generateChartData();

  // Camera feed data - using CCTV footage
  const cameraFeeds = [
    { id: 1, name: 'Camera 1 - Entrance', status: 'online', videoUrl: '/videos/cameras/14698511_1920_1080_60fps.mp4' },
    { id: 2, name: 'Camera 2 - Parking Lot', status: 'online', videoUrl: '/videos/cameras/New York 1956 42nd St WebCam - LIVE.mp4' },
    { id: 3, name: 'Camera 3 - Lobby', status: 'online', videoUrl: '/videos/cameras/5108891-uhd_3840_2160_30fps.mp4' },
    { id: 4, name: 'Camera 4 - Hallway', status: 'online', videoUrl: '/videos/cameras/2697636-uhd_1920_1440_30fps.mp4' },
    { id: 5, name: 'Camera 5 - Exit', status: 'online', videoUrl: '/videos/cameras/5977704-hd_1366_586_30fps.mp4' },
  ];

  return (
    <div className="p-8">
      {/* Camera Grid Modal */}
      {showCameraGrid && (
        <div
          className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4"
          onClick={() => setShowCameraGrid(false)}
        >
          <div
            className="bg-[var(--card-bg)] border border-[var(--border)] rounded-lg w-full max-w-7xl max-h-[90vh] overflow-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-6">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-2xl font-bold text-[var(--text-primary)]">Live Camera Feeds</h2>
                <Button
                  onClick={() => setShowCameraGrid(false)}
                  variant="outline"
                  className="border-[var(--border)] text-[var(--text-primary)] hover:bg-red-500/20"
                >
                  <X className="h-5 w-5" />
                </Button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {cameraFeeds.map((camera) => (
                  <div key={camera.id} className="border border-[var(--border)] rounded-lg overflow-hidden">
                    <div className="bg-black aspect-video relative">
                      <video
                        className="w-full h-full object-cover"
                        autoPlay
                        loop
                        muted
                        playsInline
                      >
                        <source src={camera.videoUrl} type="video/mp4" />
                        Your browser does not support the video tag.
                      </video>
                      <div className="absolute top-2 left-2 bg-black/70 px-3 py-1 rounded-full flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                        <span className="text-xs text-white font-medium">LIVE</span>
                      </div>
                    </div>
                    <div className="p-3 bg-[var(--card-bg)]">
                      <p className="text-sm font-semibold text-[var(--text-primary)]">{camera.name}</p>
                      <p className="text-xs text-[var(--text-secondary)] mt-1">Status: {camera.status}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
      {/* Header */}
      <div className="mb-8 flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-[var(--text-primary)] mb-2">
            Dashboard
          </h1>
        </div>
        <Button className="bg-[var(--accent-blue)] hover:bg-blue-600 text-white px-6 py-6 text-lg font-semibold shadow-lg shadow-blue-500/30">
          <MessageSquare className="h-6 w-6 mr-3" />
          AI Assistant
        </Button>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {stats.map((stat, index) => {
          const Icon = stat.icon;
          return (
            <Card key={index} className="border-[var(--border)] bg-[var(--card-bg)]">
              <CardContent className="pt-6">
                <div className="flex justify-between items-start mb-4">
                  <div>
                    <p className="text-sm text-[var(--text-secondary)] mb-2">
                      {stat.title}
                    </p>
                    <div className="flex items-baseline gap-2">
                      <span className="text-3xl font-bold text-[var(--text-primary)]">
                        {stat.value}
                      </span>
                      {stat.subtitle && (
                        <span className="text-sm text-[var(--text-secondary)]">
                          {stat.subtitle}
                        </span>
                      )}
                    </div>
                  </div>
                  <div className={`${stat.bgColor} p-3 rounded-lg`}>
                    <Icon className={`h-5 w-5 ${stat.iconColor}`} />
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* System Performance Metrics */}
      <Card className="border-[var(--border)] bg-[var(--card-bg)] mb-8">
        <CardHeader>
          <CardTitle className="text-xl font-bold text-[var(--text-primary)] flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            System Performance
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            <div>
              <div className="flex justify-between mb-3">
                <span className="text-base font-medium text-[var(--text-secondary)]">Detection Accuracy</span>
                <span className="text-base font-semibold text-[var(--text-primary)]">94.5%</span>
              </div>
              <div className="w-full bg-[#1e293b] rounded-full h-3">
                <div className="bg-green-500 h-3 rounded-full transition-all duration-300" style={{ width: '94.5%' }} />
              </div>
            </div>

            <div>
              <div className="flex justify-between mb-3">
                <span className="text-base font-medium text-[var(--text-secondary)]">Camera Uptime</span>
                <span className="text-base font-semibold text-[var(--text-primary)]">99.2%</span>
              </div>
              <div className="w-full bg-[#1e293b] rounded-full h-3">
                <div className="bg-blue-500 h-3 rounded-full transition-all duration-300" style={{ width: '99.2%' }} />
              </div>
            </div>

            <div>
              <div className="flex justify-between mb-3">
                <span className="text-base font-medium text-[var(--text-secondary)]">Response Time</span>
                <span className="text-base font-semibold text-[var(--text-primary)]">0.3s avg</span>
              </div>
              <div className="w-full bg-[#1e293b] rounded-full h-3">
                <div className="bg-purple-500 h-3 rounded-full transition-all duration-300" style={{ width: '85%' }} />
              </div>
            </div>

            <div>
              <div className="flex justify-between mb-3">
                <span className="text-base font-medium text-[var(--text-secondary)]">Storage Used</span>
                <span className="text-base font-semibold text-[var(--text-primary)]">67%</span>
              </div>
              <div className="w-full bg-[#1e293b] rounded-full h-3">
                <div className="bg-orange-500 h-3 rounded-full transition-all duration-300" style={{ width: '67%' }} />
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* System Performance */}
      <Card className="border-[var(--border)] bg-[var(--card-bg)] mb-8">
        <CardHeader>
          <CardTitle className="text-[var(--text-primary)]">
            System Performance (Last 24 Hours)
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(59, 130, 246, 0.2)" />
                <XAxis
                  dataKey="time"
                  stroke="#94a3b8"
                  tick={{ fill: '#94a3b8', fontSize: 12 }}
                  interval={2}
                />
                <YAxis
                  stroke="#94a3b8"
                  tick={{ fill: '#94a3b8', fontSize: 12 }}
                  label={{ value: 'Incidents', angle: -90, position: 'insideLeft', fill: '#94a3b8' }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1e293b',
                    border: '1px solid rgba(59, 130, 246, 0.3)',
                    borderRadius: '8px',
                    color: '#e2e8f0',
                    fontSize: '14px',
                    padding: '12px'
                  }}
                  labelStyle={{ color: '#e2e8f0', fontSize: '14px', fontWeight: '600' }}
                  itemStyle={{ color: '#e2e8f0', fontSize: '14px' }}
                />
                <Line
                  type="monotone"
                  dataKey="incidents"
                  stroke="#60a5fa"
                  strokeWidth={2}
                  dot={{ fill: '#60a5fa', r: 4 }}
                  activeDot={{ r: 6, fill: '#3b82f6' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Camera Status Grid */}
      <Card className="border-[var(--border)] bg-[var(--card-bg)]">
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-[var(--text-primary)]">Camera Status Grid</CardTitle>
          <Button
            className="bg-[var(--accent-blue)] hover:bg-blue-600 text-white px-4 py-2 font-semibold shadow-md shadow-blue-500/30"
            onClick={() => setShowCameraGrid(true)}
          >
            <Video className="h-4 w-4 mr-2" />
            Expand
          </Button>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2 items-center">
            {[...Array(5)].map((_, i) => (
              <div
                key={i}
                className="w-8 h-8 rounded-full bg-green-500 flex items-center justify-center"
              >
                <div className="w-6 h-6 rounded-full bg-green-600" />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
