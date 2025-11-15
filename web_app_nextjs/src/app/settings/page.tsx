import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Settings, User, Bell, Shield, Database, Palette } from 'lucide-react';
import { Button } from '@/components/ui/button';

export default function SettingsPage() {
  const settingSections = [
    {
      icon: User,
      title: 'Account Settings',
      description: 'Manage your account information and preferences',
      items: [
        { label: 'Profile Information', value: 'Update name, email, and avatar' },
        { label: 'Password', value: 'Change your password' },
        { label: 'Two-Factor Authentication', value: 'Enabled' },
      ],
    },
    {
      icon: Bell,
      title: 'Notifications',
      description: 'Configure alert and notification preferences',
      items: [
        { label: 'Email Notifications', value: 'Enabled for critical alerts' },
        { label: 'SMS Alerts', value: 'Disabled' },
        { label: 'Desktop Notifications', value: 'Enabled' },
        { label: 'Alert Threshold', value: 'Medium and above' },
      ],
    },
    {
      icon: Shield,
      title: 'Security',
      description: 'Security and privacy settings',
      items: [
        { label: 'API Keys', value: '3 active keys' },
        { label: 'Session Timeout', value: '30 minutes' },
        { label: 'IP Whitelist', value: '5 addresses configured' },
        { label: 'Audit Logs', value: 'Enabled' },
      ],
    },
    {
      icon: Database,
      title: 'Data & Storage',
      description: 'Manage data retention and storage preferences',
      items: [
        { label: 'Video Retention', value: '30 days' },
        { label: 'Storage Used', value: '2.4 TB / 5 TB' },
        { label: 'Backup Schedule', value: 'Daily at 2:00 AM' },
        { label: 'Export Format', value: 'MP4 / JSON' },
      ],
    },
  ];

  return (
    <div className="p-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-[var(--text-primary)] mb-2">
          Settings
        </h1>
        <p className="text-[var(--text-secondary)]">
          Configure your system preferences and account settings
        </p>
      </div>

      {/* Settings Sections */}
      <div className="space-y-6">
        {settingSections.map((section) => {
          const Icon = section.icon;
          return (
            <Card key={section.title} className="border-[var(--border)] bg-[var(--card-bg)]">
              <CardHeader>
                <div className="flex items-center gap-3 mb-2">
                  <div className="p-2 rounded-lg bg-[var(--accent-blue)]/10">
                    <Icon className="h-5 w-5 text-[var(--accent-blue)]" />
                  </div>
                  <div>
                    <CardTitle className="text-[var(--text-primary)]">
                      {section.title}
                    </CardTitle>
                    <CardDescription className="mt-1">
                      {section.description}
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {section.items.map((item, index) => (
                    <div key={index} className="flex items-center justify-between py-3 border-b border-[var(--border)] last:border-0">
                      <div>
                        <p className="text-sm font-medium text-[var(--text-primary)]">
                          {item.label}
                        </p>
                        <p className="text-xs text-[var(--text-secondary)] mt-1">
                          {item.value}
                        </p>
                      </div>
                      <Button variant="outline" size="sm">
                        Edit
                      </Button>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          );
        })}

        {/* System Settings */}
        <Card className="border-[var(--border)] bg-[var(--card-bg)]">
          <CardHeader>
            <div className="flex items-center gap-3 mb-2">
              <div className="p-2 rounded-lg bg-[var(--accent-blue)]/10">
                <Settings className="h-5 w-5 text-[var(--accent-blue)]" />
              </div>
              <div>
                <CardTitle className="text-[var(--text-primary)]">
                  System Configuration
                </CardTitle>
                <CardDescription className="mt-1">
                  Advanced system and detection settings
                </CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between py-3 border-b border-[var(--border)]">
                <div>
                  <p className="text-sm font-medium text-[var(--text-primary)]">
                    Detection Sensitivity
                  </p>
                  <p className="text-xs text-[var(--text-secondary)] mt-1">
                    Current: Medium (recommended)
                  </p>
                </div>
                <Button variant="outline" size="sm">
                  Configure
                </Button>
              </div>

              <div className="flex items-center justify-between py-3 border-b border-[var(--border)]">
                <div>
                  <p className="text-sm font-medium text-[var(--text-primary)]">
                    ML Model Version
                  </p>
                  <p className="text-xs text-[var(--text-secondary)] mt-1">
                    v2.4.1 - ResNet50V2 + Bi-LSTM
                  </p>
                </div>
                <Button variant="outline" size="sm">
                  Update
                </Button>
              </div>

              <div className="flex items-center justify-between py-3">
                <div>
                  <p className="text-sm font-medium text-[var(--text-primary)]">
                    System Logs
                  </p>
                  <p className="text-xs text-[var(--text-secondary)] mt-1">
                    View and export system activity logs
                  </p>
                </div>
                <Button variant="outline" size="sm">
                  View Logs
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Appearance */}
        <Card className="border-[var(--border)] bg-[var(--card-bg)]">
          <CardHeader>
            <div className="flex items-center gap-3 mb-2">
              <div className="p-2 rounded-lg bg-[var(--accent-blue)]/10">
                <Palette className="h-5 w-5 text-[var(--accent-blue)]" />
              </div>
              <div>
                <CardTitle className="text-[var(--text-primary)]">
                  Appearance
                </CardTitle>
                <CardDescription className="mt-1">
                  Customize the look and feel of your dashboard
                </CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-[var(--text-primary)]">
                  Theme
                </p>
                <p className="text-xs text-[var(--text-secondary)] mt-1">
                  Current: Dark mode
                </p>
              </div>
              <Button variant="outline" size="sm">
                Change Theme
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Save Button */}
      <div className="flex justify-end gap-3 mt-8">
        <Button variant="outline">
          Reset to Defaults
        </Button>
        <Button className="bg-[var(--accent-blue)] hover:bg-blue-600">
          Save Changes
        </Button>
      </div>
    </div>
  );
}
