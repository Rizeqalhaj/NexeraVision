'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { BarChart3, TrendingUp, FileText, Download, Calendar } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

export default function AnalysisPage() {
  const [duration, setDuration] = useState('7days');
  const [chartData, setChartData] = useState<any[]>([]);
  const [showCustomRange, setShowCustomRange] = useState(false);
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [selectedReport, setSelectedReport] = useState<any>(null);
  const [showReportModal, setShowReportModal] = useState(false);

  const reports = [
    { date: '2025-11-14', incidents: 12, cameras: 24, resolved: 11 },
    { date: '2025-11-13', incidents: 8, cameras: 24, resolved: 7 },
    { date: '2025-11-12', incidents: 15, cameras: 23, resolved: 14 },
    { date: '2025-11-11', incidents: 6, cameras: 23, resolved: 6 },
  ];

  // Camera performance data for bar chart - by location
  const locationData = [
    { location: 'Entrance', incidents: 45, efficiency: 94 },
    { location: 'Parking Lot', incidents: 32, efficiency: 91 },
    { location: 'Lobby', incidents: 28, efficiency: 96 },
    { location: 'Hallway', incidents: 15, efficiency: 88 },
    { location: 'Exit', incidents: 22, efficiency: 93 },
  ];

  // Camera performance data for table - Multiple cameras per location
  const cameraData = [
    { location: 'Building A - Floor 1 Entrance', incidents: 28, efficiency: 95 },
    { location: 'Building A - Floor 2 Entrance', incidents: 17, efficiency: 93 },
    { location: 'Building B - Floor 1 Parking Lot', incidents: 19, efficiency: 92 },
    { location: 'Building B - Floor 2 Parking Lot', incidents: 13, efficiency: 90 },
    { location: 'Building A - Floor 1 Lobby', incidents: 16, efficiency: 97 },
    { location: 'Building C - Floor 1 Lobby', incidents: 12, efficiency: 95 },
    { location: 'Building A - Floor 3 Hallway', incidents: 9, efficiency: 89 },
    { location: 'Building B - Floor 2 Hallway', incidents: 6, efficiency: 87 },
    { location: 'Building A - Floor 1 Exit', incidents: 14, efficiency: 94 },
    { location: 'Building C - Floor 2 Exit', incidents: 8, efficiency: 92 },
  ];

  const pieData = [
    { name: 'Entrance', value: 45 },
    { name: 'Parking Lot', value: 32 },
    { name: 'Lobby', value: 28 },
    { name: 'Hallway', value: 15 },
    { name: 'Exit', value: 22 },
  ];

  const COLORS = ['#60a5fa', '#22c55e', '#f59e0b', '#ef4444', '#a855f7'];

  // Generate random chart data based on duration
  const generateChartData = (selectedDuration: string, customStart?: string, customEnd?: string) => {
    const data = [];
    let days = 7;
    let labelFormat = 'day';
    let startDateTime: Date;
    let endDateTime: Date;

    if (selectedDuration === 'custom' && customStart && customEnd) {
      startDateTime = new Date(customStart);
      endDateTime = new Date(customEnd);
      days = Math.ceil((endDateTime.getTime() - startDateTime.getTime()) / (1000 * 60 * 60 * 24));

      if (days <= 31) {
        labelFormat = 'day';
      } else if (days <= 180) {
        labelFormat = 'week';
      } else {
        labelFormat = 'month';
      }
    } else {
      endDateTime = new Date();
      switch (selectedDuration) {
        case '7days':
          days = 7;
          labelFormat = 'day';
          break;
        case '30days':
          days = 30;
          labelFormat = 'day';
          break;
        case '3months':
          days = 90;
          labelFormat = 'week';
          break;
        case '6months':
          days = 180;
          labelFormat = 'month';
          break;
        case '1year':
          days = 365;
          labelFormat = 'month';
          break;
      }
      startDateTime = new Date(endDateTime);
      startDateTime.setDate(startDateTime.getDate() - days + 1);
    }

    if (labelFormat === 'day') {
      for (let i = 0; i < days; i++) {
        const date = new Date(startDateTime);
        date.setDate(date.getDate() + i);
        data.push({
          date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
          incidents: Math.floor(Math.random() * 15) + 3,
        });
      }
    } else if (labelFormat === 'week') {
      const weeks = Math.ceil(days / 7);
      for (let i = 0; i < weeks; i++) {
        const date = new Date(startDateTime);
        date.setDate(date.getDate() + (i * 7));
        data.push({
          date: `Week ${i + 1}`,
          incidents: Math.floor(Math.random() * 80) + 20,
        });
      }
    } else {
      const months = Math.ceil(days / 30);
      for (let i = 0; i < months; i++) {
        const date = new Date(startDateTime);
        date.setMonth(date.getMonth() + i);
        data.push({
          date: date.toLocaleDateString('en-US', { month: 'short' }),
          incidents: Math.floor(Math.random() * 300) + 100,
        });
      }
    }

    return data;
  };

  const handleDurationChange = (newDuration: string) => {
    setDuration(newDuration);
    if (newDuration === 'custom') {
      setShowCustomRange(true);
    } else {
      setShowCustomRange(false);
      setChartData(generateChartData(newDuration));
    }
  };

  const applyCustomRange = () => {
    if (startDate && endDate) {
      setChartData(generateChartData('custom', startDate, endDate));
    }
  };

  // Generate detailed incident report with written narratives
  const generateDetailedReport = (report: any) => {
    const incidents = [];
    const cameras = ['Camera 1 - Entrance', 'Camera 2 - Parking Lot', 'Camera 3 - Lobby', 'Camera 4 - Hallway', 'Camera 5 - Exit'];
    const incidentTypes = ['Violence Detected', 'Suspicious Activity', 'Possible Incident', 'Alert Triggered'];

    const narrativeTemplates = {
      resolved: [
        (camera: string, time: string, type: string, confidence: string, responseTime: string) =>
          `At ${time}, our AI system detected ${type.toLowerCase()} at ${camera} with ${confidence} confidence. Security personnel were immediately dispatched to the location. Upon arrival at the scene within ${responseTime}, the team conducted a thorough assessment. After reviewing the footage and interviewing witnesses, it was determined to be a false alarm. The area was secured and normal operations resumed. System sensitivity has been adjusted to reduce similar false positives.`,

        (camera: string, time: string, type: string, confidence: string, responseTime: string) =>
          `An incident was flagged by the AI detection system at ${time} from ${camera} showing ${type.toLowerCase()}. The system registered ${confidence} confidence in the detection. Security team responded within ${responseTime} and confirmed an actual incident requiring intervention. Local authorities were notified and arrived on scene. The situation was resolved with all parties safely removed from the premises. A full incident report has been filed with law enforcement.`,

        (camera: string, time: string, type: string, confidence: string, responseTime: string) =>
          `${camera} triggered an alert at ${time} for ${type.toLowerCase()} with ${confidence} confidence level. On-site security personnel investigated the alert with a response time of ${responseTime}. The incident was confirmed as a minor altercation between two individuals. Security intervened immediately, de-escalated the situation, and separated the parties involved. Both individuals were escorted off the premises. No injuries were reported and no further action required.`,

        (camera: string, time: string, type: string, confidence: string, responseTime: string) =>
          `Detection alert received at ${time} from ${camera} indicating ${type.toLowerCase()} (${confidence} confidence). Security team deployed within ${responseTime} to investigate. Upon review of live and recorded footage, it was determined that unusual lighting conditions and shadow movements triggered the alert. The incident has been classified as a false positive. Environmental factors have been noted for system calibration improvements.`,
      ],
      pending: [
        (camera: string, time: string, type: string, confidence: string) =>
          `Alert generated at ${time} from ${camera} showing ${type.toLowerCase()} with ${confidence} confidence. The incident is currently under investigation by the security team. Initial footage review is in progress. Additional camera angles are being analyzed to gather comprehensive information about the event. A determination on the nature and severity of the incident will be made pending completion of the investigation.`,

        (camera: string, time: string, type: string, confidence: string) =>
          `At ${time}, ${camera} detected ${type.toLowerCase()} with ${confidence} confidence rating. The alert has been logged and queued for review by security personnel. Due to current operational priorities, this incident is awaiting detailed investigation. Video footage has been preserved and flagged for analysis. Updates will be provided once the review is completed.`,

        (camera: string, time: string, type: string, confidence: string) =>
          `AI system flagged potential ${type.toLowerCase()} at ${camera} occurring at ${time}. The detection confidence level was ${confidence}. The incident requires further investigation to determine if intervention is necessary. Security supervisor has been notified and will conduct a comprehensive review of the footage. Status updates pending completion of preliminary assessment.`,
      ]
    };

    for (let i = 0; i < report.incidents; i++) {
      const isResolved = i < report.resolved;
      const hour = Math.floor(Math.random() * 24);
      const minute = Math.floor(Math.random() * 60);
      const confidence = Math.floor(Math.random() * 30) + 70;
      const camera = cameras[Math.floor(Math.random() * cameras.length)];
      const type = incidentTypes[Math.floor(Math.random() * incidentTypes.length)];
      const time = `${hour.toString().padStart(2, '0')}:${minute.toString().padStart(2, '0')}`;
      const responseTime = `${Math.floor(Math.random() * 10) + 1} min`;

      let narrative: string;
      if (isResolved) {
        const templates = narrativeTemplates.resolved;
        const template = templates[Math.floor(Math.random() * templates.length)];
        narrative = template(camera, time, type, `${confidence}%`, responseTime);
      } else {
        const templates = narrativeTemplates.pending;
        const template = templates[Math.floor(Math.random() * templates.length)];
        narrative = template(camera, time, type, `${confidence}%`);
      }

      incidents.push({
        id: i + 1,
        time,
        camera,
        type,
        confidence: `${confidence}%`,
        resolved: isResolved,
        responseTime: isResolved ? responseTime : 'N/A',
        narrative,
      });
    }

    return incidents.sort((a, b) => a.time.localeCompare(b.time));
  };

  const handleViewReport = (report: any) => {
    setSelectedReport({
      ...report,
      incidents: generateDetailedReport(report),
    });
    setShowReportModal(true);
  };

  useEffect(() => {
    if (duration !== 'custom') {
      setChartData(generateChartData(duration));
    }
  }, [duration]);

  return (
    <div className="p-8">
      {/* Report Modal */}
      {showReportModal && selectedReport && (
        <div
          className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4"
          onClick={() => setShowReportModal(false)}
        >
          <div
            className="bg-[var(--card-bg)] border border-[var(--border)] rounded-lg w-full max-w-5xl max-h-[90vh] overflow-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-6">
              <div className="flex justify-between items-center mb-6">
                <div>
                  <h2 className="text-2xl font-bold text-[var(--text-primary)]">
                    Incident Report
                  </h2>
                  <p className="text-[var(--text-secondary)] mt-1">
                    {new Date(selectedReport.date).toLocaleDateString('en-US', {
                      weekday: 'long',
                      year: 'numeric',
                      month: 'long',
                      day: 'numeric'
                    })}
                  </p>
                </div>
                <Button
                  onClick={() => setShowReportModal(false)}
                  variant="outline"
                  className="border-[var(--border)] text-[var(--text-primary)]"
                >
                  Close
                </Button>
              </div>

              {/* Summary */}
              <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="bg-[#1a2942] p-4 rounded-lg">
                  <p className="text-sm text-[var(--text-secondary)]">Total Incidents</p>
                  <p className="text-2xl font-bold text-[var(--text-primary)]">{selectedReport.incidents.length}</p>
                </div>
                <div className="bg-[#1a2942] p-4 rounded-lg">
                  <p className="text-sm text-[var(--text-secondary)]">Resolved</p>
                  <p className="text-2xl font-bold text-green-500">{selectedReport.resolved}</p>
                </div>
                <div className="bg-[#1a2942] p-4 rounded-lg">
                  <p className="text-sm text-[var(--text-secondary)]">Pending</p>
                  <p className="text-2xl font-bold text-yellow-500">{selectedReport.incidents.length - selectedReport.resolved}</p>
                </div>
              </div>

              {/* Incident Details */}
              <div className="space-y-3">
                <h3 className="text-lg font-semibold text-[var(--text-primary)] mb-4">Incident Details</h3>
                {selectedReport.incidents.map((incident: any) => (
                  <div
                    key={incident.id}
                    className="border border-[var(--border)] rounded-lg p-4 bg-[#0a1929]"
                  >
                    <div className="flex justify-between items-start mb-3">
                      <div>
                        <div className="flex items-center gap-3">
                          <span className={`w-3 h-3 rounded-full ${
                            incident.resolved ? 'bg-green-500' : 'bg-yellow-500'
                          }`} />
                          <span className="font-semibold text-[var(--text-primary)]">
                            Incident #{incident.id}
                          </span>
                          <span className="text-sm text-[var(--text-secondary)]">
                            {incident.time}
                          </span>
                        </div>
                      </div>
                      <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                        incident.resolved
                          ? 'bg-green-500/20 text-green-500'
                          : 'bg-yellow-500/20 text-yellow-500'
                      }`}>
                        {incident.resolved ? 'Resolved' : 'Pending'}
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-[var(--text-secondary)]">Camera</p>
                        <p className="text-[var(--text-primary)] font-medium">{incident.camera}</p>
                      </div>
                      <div>
                        <p className="text-[var(--text-secondary)]">Type</p>
                        <p className="text-[var(--text-primary)] font-medium">{incident.type}</p>
                      </div>
                      <div>
                        <p className="text-[var(--text-secondary)]">Confidence</p>
                        <p className="text-[var(--text-primary)] font-medium">{incident.confidence}</p>
                      </div>
                      <div>
                        <p className="text-[var(--text-secondary)]">Response Time</p>
                        <p className="text-[var(--text-primary)] font-medium">{incident.responseTime}</p>
                      </div>
                    </div>

                    <div className="mt-3 pt-3 border-t border-[var(--border)]">
                      <p className="text-[var(--text-secondary)] text-sm mb-2 font-semibold">INCIDENT REPORT</p>
                      <p className="text-[var(--text-primary)] text-base leading-relaxed">{incident.narrative}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-[var(--text-primary)] mb-2">
          Analysis & Reports
        </h1>
        <p className="text-[var(--text-secondary)]">
          Detailed analytics and incident reports
        </p>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <Card className="border-[var(--border)] bg-[var(--card-bg)]">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-[var(--text-secondary)]">
              Total Incidents (30d)
            </CardTitle>
            <BarChart3 className="h-4 w-4 text-blue-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-[var(--text-primary)]">342</div>
            <p className="text-xs text-green-500 mt-1">-8% from last month</p>
          </CardContent>
        </Card>

        <Card className="border-[var(--border)] bg-[var(--card-bg)]">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-[var(--text-secondary)]">
              Average Response
            </CardTitle>
            <TrendingUp className="h-4 w-4 text-green-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-[var(--text-primary)]">2.3 min</div>
            <p className="text-xs text-green-500 mt-1">12% faster</p>
          </CardContent>
        </Card>

        <Card className="border-[var(--border)] bg-[var(--card-bg)]">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-[var(--text-secondary)]">
              Resolution Rate
            </CardTitle>
            <FileText className="h-4 w-4 text-purple-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-[var(--text-primary)]">96.2%</div>
            <p className="text-xs text-green-500 mt-1">+2.1% improvement</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Incident Trends Chart */}
        <Card className="border-[var(--border)] bg-[var(--card-bg)]">
          <CardHeader>
            <div className="flex justify-between items-start gap-4">
              <div>
                <CardTitle className="text-[var(--text-primary)]">
                  Incident Trends
                </CardTitle>
                <CardDescription>
                  Daily incident detection over selected period
                </CardDescription>
              </div>
              <div className="flex flex-col gap-2">
                <select
                  value={duration}
                  onChange={(e) => handleDurationChange(e.target.value)}
                  className="bg-[#1a2942] border border-[var(--border)] text-[var(--text-primary)] px-3 py-2 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-[var(--accent-blue)]"
                >
                  <option value="7days">Last 7 Days</option>
                  <option value="30days">Last 30 Days</option>
                  <option value="3months">Last 3 Months</option>
                  <option value="6months">Last 6 Months</option>
                  <option value="1year">Last Year</option>
                  <option value="custom">Custom Range</option>
                </select>

                {showCustomRange && (
                  <div className="flex flex-col gap-2 p-3 border border-[var(--border)] rounded-md bg-[#0a1929]">
                    <div className="flex flex-col gap-1">
                      <label className="text-xs text-[var(--text-secondary)]">Start Date</label>
                      <input
                        type="date"
                        value={startDate}
                        onChange={(e) => setStartDate(e.target.value)}
                        className="bg-[#1a2942] border border-[var(--border)] text-[var(--text-primary)] px-2 py-1 rounded text-sm focus:outline-none focus:ring-2 focus:ring-[var(--accent-blue)]"
                      />
                    </div>
                    <div className="flex flex-col gap-1">
                      <label className="text-xs text-[var(--text-secondary)]">End Date</label>
                      <input
                        type="date"
                        value={endDate}
                        onChange={(e) => setEndDate(e.target.value)}
                        className="bg-[#1a2942] border border-[var(--border)] text-[var(--text-primary)] px-2 py-1 rounded text-sm focus:outline-none focus:ring-2 focus:ring-[var(--accent-blue)]"
                      />
                    </div>
                    <Button
                      onClick={applyCustomRange}
                      size="sm"
                      className="bg-[var(--accent-blue)] hover:bg-blue-600 text-white text-xs"
                    >
                      Apply
                    </Button>
                  </div>
                )}
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(59, 130, 246, 0.2)" />
                  <XAxis
                    dataKey="date"
                    stroke="#94a3b8"
                    tick={{ fill: '#94a3b8', fontSize: 11 }}
                    angle={-45}
                    textAnchor="end"
                    height={70}
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
                    dot={{ fill: '#60a5fa', r: 3 }}
                    activeDot={{ r: 5, fill: '#3b82f6' }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Reports */}
        <Card className="border-[var(--border)] bg-[var(--card-bg)]">
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle className="text-[var(--text-primary)]">
                Reports
              </CardTitle>
              <CardDescription>
                Last 4 days summary
              </CardDescription>
            </div>
            <Button variant="outline" size="sm" className="gap-2 border-[var(--border)] text-[var(--text-primary)]">
              <Download className="h-4 w-4" />
              Export
            </Button>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {reports.map((report) => (
                <div key={report.date} className="flex items-center justify-between p-3 border border-[var(--border)] rounded-lg">
                  <div className="flex items-center gap-3">
                    <Calendar className="h-5 w-5 text-[var(--text-secondary)]" />
                    <div>
                      <p className="text-sm font-medium text-[var(--text-primary)]">
                        {new Date(report.date).toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })}
                      </p>
                      <p className="text-xs text-[var(--text-secondary)]">
                        {report.incidents} incidents • {report.resolved} resolved
                      </p>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-[var(--accent-blue)] hover:text-blue-400"
                    onClick={() => handleViewReport(report)}
                  >
                    View
                  </Button>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Additional Analytics Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
        {/* Bar Chart - Incidents by Camera */}
        <Card className="border-[var(--border)] bg-[var(--card-bg)]">
          <CardHeader>
            <CardTitle className="text-[var(--text-primary)]">
              Incidents by Camera Location
            </CardTitle>
            <CardDescription>
              Total incidents detected per camera
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={locationData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(59, 130, 246, 0.2)" />
                  <XAxis
                    dataKey="location"
                    stroke="#94a3b8"
                    tick={{ fill: '#94a3b8', fontSize: 12 }}
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
                  <Bar dataKey="incidents" fill="#60a5fa" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Pie Chart - Distribution */}
        <Card className="border-[var(--border)] bg-[var(--card-bg)]">
          <CardHeader>
            <CardTitle className="text-[var(--text-primary)]">
              Detection Distribution
            </CardTitle>
            <CardDescription>
              Percentage breakdown by location
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name} ${((percent || 0) * 100).toFixed(0)}%`}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1e293b',
                      border: '1px solid rgba(59, 130, 246, 0.3)',
                      borderRadius: '8px',
                      color: '#e2e8f0',
                      fontSize: '14px',
                      padding: '12px'
                    }}
                  />
                  <Legend
                    wrapperStyle={{
                      fontSize: '14px',
                      color: '#e2e8f0'
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Camera Efficiency Table */}
      <Card className="border-[var(--border)] bg-[var(--card-bg)] mt-6">
        <CardHeader>
          <CardTitle className="text-[var(--text-primary)]">
            Camera Detection Efficiency
          </CardTitle>
          <CardDescription>
            Detection accuracy and performance metrics by location
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-[var(--border)]">
                  <th className="text-left py-3 px-4 text-[var(--text-secondary)] font-semibold text-sm">Location</th>
                  <th className="text-left py-3 px-4 text-[var(--text-secondary)] font-semibold text-sm">Incidents</th>
                  <th className="text-left py-3 px-4 text-[var(--text-secondary)] font-semibold text-sm">Efficiency</th>
                  <th className="text-left py-3 px-4 text-[var(--text-secondary)] font-semibold text-sm">Status</th>
                </tr>
              </thead>
              <tbody>
                {cameraData.map((camera, index) => (
                  <tr key={index} className="border-b border-[var(--border)] hover:bg-[#0a1929] transition-colors">
                    <td className="py-3 px-4 text-[var(--text-primary)] font-medium">{camera.location}</td>
                    <td className="py-3 px-4 text-[var(--text-primary)]">{camera.incidents}</td>
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <div className="w-24 bg-[#1e293b] rounded-full h-2">
                          <div
                            className="bg-green-500 h-2 rounded-full transition-all duration-300"
                            style={{ width: `${camera.efficiency}%` }}
                          />
                        </div>
                        <span className="text-[var(--text-primary)] text-sm font-semibold">{camera.efficiency}%</span>
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                        camera.efficiency >= 95 ? 'bg-green-500/20 text-green-500' :
                        camera.efficiency >= 90 ? 'bg-blue-500/20 text-blue-500' :
                        'bg-yellow-500/20 text-yellow-500'
                      }`}>
                        {camera.efficiency >= 95 ? 'Excellent' : camera.efficiency >= 90 ? 'Good' : 'Fair'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
