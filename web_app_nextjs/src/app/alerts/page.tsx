import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Bell, AlertTriangle, CheckCircle, Clock, Filter } from 'lucide-react';
import { Button } from '@/components/ui/button';

export default function AlertsPage() {
  const alerts = [
    {
      id: 1,
      severity: 'high',
      title: 'Violence Detected',
      camera: 'Entrance Main - Camera 01',
      time: '2 minutes ago',
      status: 'active',
      description: 'Potential violent behavior detected in main entrance area',
    },
    {
      id: 2,
      severity: 'medium',
      title: 'Suspicious Activity',
      camera: 'Parking Lot - Camera 12',
      time: '15 minutes ago',
      status: 'investigating',
      description: 'Unusual movement patterns detected',
    },
    {
      id: 3,
      severity: 'high',
      title: 'Violence Detected',
      camera: 'Lobby - Camera 18',
      time: '1 hour ago',
      status: 'resolved',
      description: 'Physical altercation detected and resolved',
    },
    {
      id: 4,
      severity: 'low',
      title: 'Camera Offline',
      camera: 'Cafeteria - Camera 08',
      time: '2 hours ago',
      status: 'resolved',
      description: 'Camera connection restored',
    },
  ];

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high': return 'text-red-500 bg-red-500/10 border-red-500/20';
      case 'medium': return 'text-orange-500 bg-orange-500/10 border-orange-500/20';
      case 'low': return 'text-blue-500 bg-blue-500/10 border-blue-500/20';
      default: return 'text-gray-500 bg-gray-500/10 border-gray-500/20';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-red-500 bg-red-500/10';
      case 'investigating': return 'text-orange-500 bg-orange-500/10';
      case 'resolved': return 'text-green-500 bg-green-500/10';
      default: return 'text-gray-500 bg-gray-500/10';
    }
  };

  return (
    <div className="p-8">
      {/* Header */}
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-3xl font-bold text-[var(--text-primary)] mb-2">
            Alerts & Notifications
          </h1>
          <p className="text-[var(--text-secondary)]">
            Monitor and manage security alerts in real-time
          </p>
        </div>
        <Button variant="outline" className="gap-2">
          <Filter className="h-4 w-4" />
          Filter
        </Button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <Card className="border-[var(--border)] bg-[var(--card-bg)]">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-[var(--text-secondary)]">
              Active Alerts
            </CardTitle>
            <Bell className="h-4 w-4 text-red-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-[var(--text-primary)]">3</div>
            <p className="text-xs text-[var(--text-secondary)] mt-1">Requires attention</p>
          </CardContent>
        </Card>

        <Card className="border-[var(--border)] bg-[var(--card-bg)]">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-[var(--text-secondary)]">
              Investigating
            </CardTitle>
            <Clock className="h-4 w-4 text-orange-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-[var(--text-primary)]">5</div>
            <p className="text-xs text-[var(--text-secondary)] mt-1">In progress</p>
          </CardContent>
        </Card>

        <Card className="border-[var(--border)] bg-[var(--card-bg)]">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-[var(--text-secondary)]">
              Resolved Today
            </CardTitle>
            <CheckCircle className="h-4 w-4 text-green-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-[var(--text-primary)]">12</div>
            <p className="text-xs text-green-500 mt-1">+2 from yesterday</p>
          </CardContent>
        </Card>

        <Card className="border-[var(--border)] bg-[var(--card-bg)]">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-[var(--text-secondary)]">
              Avg Response
            </CardTitle>
            <AlertTriangle className="h-4 w-4 text-blue-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-[var(--text-primary)]">2.3m</div>
            <p className="text-xs text-[var(--text-secondary)] mt-1">Per incident</p>
          </CardContent>
        </Card>
      </div>

      {/* Alerts List */}
      <Card className="border-[var(--border)] bg-[var(--card-bg)]">
        <CardHeader>
          <CardTitle className="text-[var(--text-primary)]">Recent Alerts</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {alerts.map((alert) => (
              <div key={alert.id} className="p-4 border border-[var(--border)] rounded-lg">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-start gap-3 flex-1">
                    {/* Severity Badge */}
                    <div className={`px-2 py-1 rounded text-xs font-medium uppercase border ${getSeverityColor(alert.severity)}`}>
                      {alert.severity}
                    </div>

                    {/* Alert Info */}
                    <div className="flex-1">
                      <h3 className="font-semibold text-[var(--text-primary)] mb-1">
                        {alert.title}
                      </h3>
                      <p className="text-sm text-[var(--text-secondary)] mb-2">
                        {alert.camera}
                      </p>
                      <p className="text-sm text-[var(--text-secondary)]">
                        {alert.description}
                      </p>
                    </div>
                  </div>

                  {/* Status & Time */}
                  <div className="text-right">
                    <div className={`inline-block px-3 py-1 rounded-full text-xs font-medium mb-2 ${getStatusColor(alert.status)}`}>
                      {alert.status}
                    </div>
                    <p className="text-xs text-[var(--text-secondary)]">{alert.time}</p>
                  </div>
                </div>

                {/* Actions */}
                <div className="flex gap-2 mt-3 pt-3 border-t border-[var(--border)]">
                  <Button variant="outline" size="sm">
                    View Details
                  </Button>
                  {alert.status === 'active' && (
                    <>
                      <Button variant="outline" size="sm">
                        Acknowledge
                      </Button>
                      <Button size="sm" className="bg-[var(--accent-blue)] hover:bg-blue-600">
                        Resolve
                      </Button>
                    </>
                  )}
                  {alert.status === 'investigating' && (
                    <Button size="sm" className="bg-[var(--accent-blue)] hover:bg-blue-600">
                      Mark Resolved
                    </Button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
