import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { BarChart3, Camera, AlertTriangle, CheckCircle, Activity, Clock } from 'lucide-react';

export default function Dashboard() {
  const stats = [
    {
      title: 'Total Cameras',
      value: '24',
      icon: Camera,
      trend: '+2 this week',
      color: 'text-blue-500',
    },
    {
      title: 'Active Detections',
      value: '3',
      icon: Activity,
      trend: 'Live monitoring',
      color: 'text-green-500',
    },
    {
      title: 'Alerts Today',
      value: '7',
      icon: AlertTriangle,
      trend: '-12% from yesterday',
      color: 'text-orange-500',
    },
    {
      title: 'Resolved',
      value: '142',
      icon: CheckCircle,
      trend: 'This month',
      color: 'text-purple-500',
    },
  ];

  const recentActivity = [
    { time: '2 min ago', event: 'Violence detected - Camera 12', status: 'warning' },
    { time: '15 min ago', event: 'All cameras operational', status: 'success' },
    { time: '1 hour ago', event: 'Alert resolved - Camera 8', status: 'success' },
    { time: '3 hours ago', event: 'New camera added - Camera 24', status: 'info' },
  ];

  return (
    <div className="p-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-[var(--text-primary)] mb-2">
          Dashboard Overview
        </h1>
        <p className="text-[var(--text-secondary)]">
          Monitor your violence detection system in real-time
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {stats.map((stat) => {
          const Icon = stat.icon;
          return (
            <Card key={stat.title} className="border-[var(--border)] bg-[var(--card-bg)]">
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-sm font-medium text-[var(--text-secondary)]">
                  {stat.title}
                </CardTitle>
                <Icon className={`h-4 w-4 ${stat.color}`} />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-[var(--text-primary)]">
                  {stat.value}
                </div>
                <p className="text-xs text-[var(--text-secondary)] mt-1">
                  {stat.trend}
                </p>
              </CardContent>
            </Card>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Activity */}
        <Card className="border-[var(--border)] bg-[var(--card-bg)]">
          <CardHeader>
            <CardTitle className="text-[var(--text-primary)] flex items-center gap-2">
              <Clock className="h-5 w-5" />
              Recent Activity
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {recentActivity.map((activity, index) => (
                <div key={index} className="flex items-start gap-3 pb-3 border-b border-[var(--border)] last:border-0">
                  <div className={`w-2 h-2 rounded-full mt-2 ${
                    activity.status === 'warning' ? 'bg-orange-500' :
                    activity.status === 'success' ? 'bg-green-500' :
                    'bg-blue-500'
                  }`} />
                  <div className="flex-1">
                    <p className="text-sm text-[var(--text-primary)]">
                      {activity.event}
                    </p>
                    <p className="text-xs text-[var(--text-secondary)] mt-1">
                      {activity.time}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Quick Stats */}
        <Card className="border-[var(--border)] bg-[var(--card-bg)]">
          <CardHeader>
            <CardTitle className="text-[var(--text-primary)] flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              System Performance
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between mb-2">
                  <span className="text-sm text-[var(--text-secondary)]">Detection Accuracy</span>
                  <span className="text-sm font-medium text-[var(--text-primary)]">94.5%</span>
                </div>
                <div className="w-full bg-[var(--border)] rounded-full h-2">
                  <div className="bg-green-500 h-2 rounded-full" style={{ width: '94.5%' }} />
                </div>
              </div>

              <div>
                <div className="flex justify-between mb-2">
                  <span className="text-sm text-[var(--text-secondary)]">Camera Uptime</span>
                  <span className="text-sm font-medium text-[var(--text-primary)]">99.2%</span>
                </div>
                <div className="w-full bg-[var(--border)] rounded-full h-2">
                  <div className="bg-blue-500 h-2 rounded-full" style={{ width: '99.2%' }} />
                </div>
              </div>

              <div>
                <div className="flex justify-between mb-2">
                  <span className="text-sm text-[var(--text-secondary)]">Response Time</span>
                  <span className="text-sm font-medium text-[var(--text-primary)]">0.3s avg</span>
                </div>
                <div className="w-full bg-[var(--border)] rounded-full h-2">
                  <div className="bg-purple-500 h-2 rounded-full" style={{ width: '85%' }} />
                </div>
              </div>

              <div>
                <div className="flex justify-between mb-2">
                  <span className="text-sm text-[var(--text-secondary)]">Storage Used</span>
                  <span className="text-sm font-medium text-[var(--text-primary)]">67%</span>
                </div>
                <div className="w-full bg-[var(--border)] rounded-full h-2">
                  <div className="bg-orange-500 h-2 rounded-full" style={{ width: '67%' }} />
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
