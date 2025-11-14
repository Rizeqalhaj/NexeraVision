import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { BarChart3, TrendingUp, FileText, Download, Calendar } from 'lucide-react';
import { Button } from '@/components/ui/button';

export default function AnalysisPage() {
  const reports = [
    { date: '2025-11-14', incidents: 12, cameras: 24, resolved: 11 },
    { date: '2025-11-13', incidents: 8, cameras: 24, resolved: 7 },
    { date: '2025-11-12', incidents: 15, cameras: 23, resolved: 14 },
    { date: '2025-11-11', incidents: 6, cameras: 23, resolved: 6 },
  ];

  return (
    <div className="p-8">
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
        {/* Charts Placeholder */}
        <Card className="border-[var(--border)] bg-[var(--card-bg)]">
          <CardHeader>
            <CardTitle className="text-[var(--text-primary)]">
              Incident Trends
            </CardTitle>
            <CardDescription>
              Daily incident detection over the past week
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64 flex items-center justify-center border-2 border-dashed border-[var(--border)] rounded-lg">
              <div className="text-center text-[var(--text-secondary)]">
                <BarChart3 className="h-12 w-12 mx-auto mb-2 opacity-50" />
                <p>Chart visualization coming soon</p>
                <p className="text-xs mt-1">Integration with charting library</p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Recent Reports */}
        <Card className="border-[var(--border)] bg-[var(--card-bg)]">
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle className="text-[var(--text-primary)]">
                Daily Reports
              </CardTitle>
              <CardDescription>
                Last 4 days summary
              </CardDescription>
            </div>
            <Button variant="outline" size="sm" className="gap-2">
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
                  <Button variant="ghost" size="sm">
                    View
                  </Button>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Additional Analytics Section */}
      <Card className="border-[var(--border)] bg-[var(--card-bg)] mt-6">
        <CardHeader>
          <CardTitle className="text-[var(--text-primary)]">
            Camera Performance Analysis
          </CardTitle>
          <CardDescription>
            Breakdown by camera location and detection efficiency
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-48 flex items-center justify-center border-2 border-dashed border-[var(--border)] rounded-lg">
            <div className="text-center text-[var(--text-secondary)]">
              <p>Detailed camera analytics coming soon</p>
              <p className="text-xs mt-1">Heatmaps, efficiency scores, and more</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
