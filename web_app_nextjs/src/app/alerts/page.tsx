'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Bell, AlertTriangle, CheckCircle, Clock, Filter, Eye, Check, CheckCheck, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useState } from 'react';

export default function AlertsPage() {
  const [selectedAlert, setSelectedAlert] = useState<any>(null);
  const [showDetailModal, setShowDetailModal] = useState(false);
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

  const generateAlertNarrative = (alert: any) => {
    const confidence = Math.floor(Math.random() * 20) + 80;
    const responseTime = `${Math.floor(Math.random() * 5) + 1} min`;

    const resolvedTemplates = [
      `At ${alert.time}, our AI system detected ${alert.title.toLowerCase()} at ${alert.camera} with ${confidence}% confidence. Security personnel were immediately dispatched to the location. Upon arrival at the scene within ${responseTime}, the team conducted a thorough assessment. After reviewing the footage and interviewing witnesses, it was determined to be a false alarm. The area was secured and normal operations resumed. System sensitivity has been adjusted to reduce similar false positives.`,

      `An incident was flagged by the AI detection system at ${alert.time} from ${alert.camera} showing ${alert.title.toLowerCase()}. The system registered ${confidence}% confidence in the detection. Security team responded within ${responseTime} and confirmed an actual incident requiring intervention. Local authorities were notified and arrived on scene. The situation was resolved with all parties safely removed from the premises. A full incident report has been filed with law enforcement.`,

      `${alert.camera} triggered an alert at ${alert.time} for ${alert.title.toLowerCase()} with ${confidence}% confidence level. On-site security personnel investigated the alert with a response time of ${responseTime}. The incident was confirmed as a minor altercation between two individuals. Security intervened immediately, de-escalated the situation, and separated the parties involved. Both individuals were escorted off the premises. No injuries were reported and no further action required.`,
    ];

    const activeTemplates = [
      `Alert generated at ${alert.time} from ${alert.camera} showing ${alert.title.toLowerCase()} with ${confidence}% confidence. The incident is currently under active response by the security team. Emergency protocols have been activated. Live monitoring is in progress with security personnel en route to the location. Priority level has been set to HIGH based on the nature of the alert.`,

      `ACTIVE ALERT: At ${alert.time}, ${alert.camera} detected ${alert.title.toLowerCase()} with ${confidence}% confidence rating. Security personnel have been dispatched and are currently responding to the incident. Real-time footage is being monitored from the command center. All relevant stakeholders have been notified and emergency procedures are in effect.`,
    ];

    const investigatingTemplates = [
      `Alert generated at ${alert.time} from ${alert.camera} showing ${alert.title.toLowerCase()} with ${confidence}% confidence. The incident is currently under investigation by the security team. Initial footage review is in progress. Additional camera angles are being analyzed to gather comprehensive information about the event. A determination on the nature and severity of the incident will be made pending completion of the investigation.`,

      `At ${alert.time}, ${alert.camera} detected ${alert.title.toLowerCase()} with ${confidence}% confidence rating. The alert has been escalated to the investigation team for detailed analysis. Security supervisor is conducting a comprehensive review of the footage and surrounding camera feeds. Preliminary assessment is underway with expected resolution timeline within the next 30 minutes.`,
    ];

    let templates;
    if (alert.status === 'resolved') {
      templates = resolvedTemplates;
    } else if (alert.status === 'active') {
      templates = activeTemplates;
    } else {
      templates = investigatingTemplates;
    }

    return templates[Math.floor(Math.random() * templates.length)];
  };

  const handleViewDetails = (alert: any) => {
    setSelectedAlert({
      ...alert,
      narrative: generateAlertNarrative(alert),
      confidence: `${Math.floor(Math.random() * 20) + 80}%`,
      responseTime: alert.status === 'resolved' ? `${Math.floor(Math.random() * 5) + 1} min` : 'N/A',
    });
    setShowDetailModal(true);
  };

  return (
    <div className="p-8">
      {/* Detail Modal */}
      {showDetailModal && selectedAlert && (
        <div
          className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4"
          onClick={() => setShowDetailModal(false)}
        >
          <div
            className="bg-[var(--card-bg)] border border-[var(--border)] rounded-lg w-full max-w-3xl max-h-[90vh] overflow-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-6">
              <div className="flex justify-between items-start mb-6">
                <div>
                  <h2 className="text-2xl font-bold text-[var(--text-primary)] mb-2">Alert Details</h2>
                  <p className="text-[var(--text-secondary)]">{selectedAlert.camera}</p>
                </div>
                <Button
                  onClick={() => setShowDetailModal(false)}
                  variant="outline"
                  className="border-[var(--border)] text-[var(--text-primary)] hover:bg-red-500/20"
                >
                  <X className="h-5 w-5" />
                </Button>
              </div>

              <div className="space-y-4">
                <div className="p-4 border border-[var(--border)] rounded-lg bg-[#0a1929]">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className={`px-3 py-1.5 rounded text-sm font-semibold uppercase border ${getSeverityColor(selectedAlert.severity)}`}>
                        {selectedAlert.severity}
                      </div>
                      <h3 className="text-xl font-bold text-[var(--text-primary)]">{selectedAlert.title}</h3>
                    </div>
                    <div className={`px-4 py-1.5 rounded-full text-sm font-semibold ${getStatusColor(selectedAlert.status)}`}>
                      {selectedAlert.status}
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4 mb-4">
                    <div>
                      <p className="text-[var(--text-secondary)] text-sm mb-1">Camera Location</p>
                      <p className="text-[var(--text-primary)] font-semibold text-base">{selectedAlert.camera}</p>
                    </div>
                    <div>
                      <p className="text-[var(--text-secondary)] text-sm mb-1">Time Detected</p>
                      <p className="text-[var(--text-primary)] font-semibold text-base">{selectedAlert.time}</p>
                    </div>
                    <div>
                      <p className="text-[var(--text-secondary)] text-sm mb-1">Detection Confidence</p>
                      <p className="text-[var(--text-primary)] font-semibold text-base">{selectedAlert.confidence}</p>
                    </div>
                    <div>
                      <p className="text-[var(--text-secondary)] text-sm mb-1">Response Time</p>
                      <p className="text-[var(--text-primary)] font-semibold text-base">{selectedAlert.responseTime}</p>
                    </div>
                  </div>

                  <div className="mt-4 pt-4 border-t border-[var(--border)]">
                    <p className="text-[var(--text-secondary)] text-sm mb-2 font-semibold">INCIDENT REPORT</p>
                    <p className="text-[var(--text-primary)] text-base leading-relaxed">{selectedAlert.narrative}</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

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
        <Button className="gap-2 bg-[#1a2942] hover:bg-[#0a1929] text-[var(--text-primary)] font-semibold border border-[var(--accent-blue)] shadow-md shadow-blue-500/20">
          <Filter className="h-4 w-4" />
          Filter Alerts
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
                  <Button
                    size="sm"
                    className="bg-[#1a2942] hover:bg-[#0a1929] text-[var(--text-primary)] font-semibold border border-[var(--accent-blue)] shadow-sm shadow-blue-500/20"
                    onClick={() => handleViewDetails(alert)}
                  >
                    <Eye className="h-3.5 w-3.5 mr-1.5" />
                    View Details
                  </Button>
                  {alert.status === 'active' && (
                    <>
                      <Button
                        size="sm"
                        className="bg-orange-600 hover:bg-orange-700 text-white font-semibold shadow-sm shadow-orange-500/30 border-0"
                      >
                        <Check className="h-3.5 w-3.5 mr-1.5" />
                        Acknowledge
                      </Button>
                      <Button
                        size="sm"
                        className="bg-green-600 hover:bg-green-700 text-white font-semibold shadow-md shadow-green-500/30 border-0"
                      >
                        <CheckCheck className="h-3.5 w-3.5 mr-1.5" />
                        Resolve
                      </Button>
                    </>
                  )}
                  {alert.status === 'investigating' && (
                    <Button
                      size="sm"
                      className="bg-green-600 hover:bg-green-700 text-white font-semibold shadow-md shadow-green-500/30 border-0"
                    >
                      <CheckCheck className="h-3.5 w-3.5 mr-1.5" />
                      Resolve
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
