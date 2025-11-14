import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Camera, Plus, Settings, Wifi, WifiOff } from 'lucide-react';
import { Button } from '@/components/ui/button';

export default function CamerasPage() {
  const cameras = [
    { id: 1, name: 'Entrance Main', location: 'Building A - Floor 1', status: 'online', alerts: 3 },
    { id: 2, name: 'Parking Lot', location: 'Outdoor - North', status: 'online', alerts: 0 },
    { id: 3, name: 'Hallway B2', location: 'Building B - Floor 2', status: 'online', alerts: 1 },
    { id: 4, name: 'Cafeteria', location: 'Building A - Floor 1', status: 'offline', alerts: 0 },
    { id: 5, name: 'Stairwell A', location: 'Building A - Floor 3', status: 'online', alerts: 0 },
    { id: 6, name: 'Lobby', location: 'Main Building', status: 'online', alerts: 2 },
  ];

  return (
    <div className="p-8">
      {/* Header */}
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-3xl font-bold text-[var(--text-primary)] mb-2">
            Camera Management
          </h1>
          <p className="text-[var(--text-secondary)]">
            Monitor and configure your security cameras
          </p>
        </div>
        <Button className="gap-2 bg-[var(--accent-blue)] hover:bg-blue-600">
          <Plus className="h-4 w-4" />
          Add Camera
        </Button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
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
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
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
              {/* Camera Preview Placeholder */}
              <div className="aspect-video bg-[var(--border)] rounded-lg mb-3 flex items-center justify-center">
                <Camera className="h-12 w-12 text-[var(--text-secondary)] opacity-50" />
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
                <Button variant="outline" size="sm" className="flex-1">
                  View Live
                </Button>
                <Button variant="outline" size="sm" className="flex-1">
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
