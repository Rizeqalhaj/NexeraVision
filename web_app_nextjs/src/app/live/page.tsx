import Link from 'next/link';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Upload, Video, Grid3x3, ArrowRight } from 'lucide-react';

export default function LiveDashboard() {
  return (
    <div className="p-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-[var(--text-primary)] mb-2">
          Live Detection
        </h1>
        <p className="text-[var(--text-secondary)]">
          Real-time violence detection tools for your CCTV system
        </p>
      </div>

      {/* Feature Cards */}
      <div className="grid md:grid-cols-3 gap-6 mb-8">
          {/* File Upload Card */}
          <Card className="border-[var(--border)] bg-[var(--card-bg)] hover:border-[var(--accent-blue)] transition-all">
            <CardHeader>
              <div className="w-12 h-12 rounded-lg bg-blue-500/10 flex items-center justify-center mb-4">
                <Upload className="h-6 w-6 text-[var(--accent-blue)]" />
              </div>
              <CardTitle className="text-[var(--text-primary)]">File Upload Detection</CardTitle>
              <CardDescription className="text-[var(--text-secondary)]">
                Upload existing CCTV footage for instant AI analysis
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Link href="/live/upload">
                <Button className="w-full bg-[var(--accent-blue)] hover:bg-blue-600">
                  Upload Video
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
            </CardContent>
          </Card>

          {/* Live Camera Card */}
          <Card className="border-[var(--border)] bg-[var(--card-bg)] hover:border-[var(--accent-blue)] transition-all">
            <CardHeader>
              <div className="w-12 h-12 rounded-lg bg-red-500/10 flex items-center justify-center mb-4">
                <Video className="h-6 w-6 text-[var(--danger-red)]" />
              </div>
              <CardTitle className="text-[var(--text-primary)]">Live Camera Detection</CardTitle>
              <CardDescription className="text-[var(--text-secondary)]">
                Real-time violence detection with webcam feed
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Link href="/live/camera">
                <Button className="w-full bg-[var(--danger-red)] hover:bg-red-600">
                  Start Live
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
            </CardContent>
          </Card>

          {/* Multi-Camera Card */}
          <Card className="border-[var(--border)] bg-[var(--card-bg)] opacity-60">
            <CardHeader>
              <div className="w-12 h-12 rounded-lg bg-green-500/10 flex items-center justify-center mb-4">
                <Grid3x3 className="h-6 w-6 text-[var(--success-green)]" />
              </div>
              <CardTitle className="text-[var(--text-primary)]">Multi-Camera Grid</CardTitle>
              <CardDescription className="text-[var(--text-secondary)]">
                Screen recording segmentation for 20-100 cameras
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button disabled className="w-full">
                Coming Soon
              </Button>
            </CardContent>
          </Card>
        </div>

      {/* Features Section */}
      <div className="mt-12">
        <h2 className="text-2xl font-bold text-[var(--text-primary)] mb-6">
          Why NexaraVision?
        </h2>
        <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <h3 className="text-xl font-semibold text-[var(--accent-blue)]">
                90-95% Accuracy
              </h3>
              <p className="text-[var(--text-secondary)]">
                Advanced AI model trained on diverse violence datasets with ResNet50V2 + Bi-LSTM architecture
              </p>
            </div>
            <div className="space-y-4">
              <h3 className="text-xl font-semibold text-[var(--accent-blue)]">
                Real-Time Detection
              </h3>
              <p className="text-[var(--text-secondary)]">
                Less than 500ms latency from camera feed to alert notification
              </p>
            </div>
            <div className="space-y-4">
              <h3 className="text-xl font-semibold text-[var(--accent-blue)]">
                Cost Effective
              </h3>
              <p className="text-[var(--text-secondary)]">
                $5-15 per camera/month vs $50-200 for enterprise solutions
              </p>
            </div>
            <div className="space-y-4">
              <h3 className="text-xl font-semibold text-[var(--accent-blue)]">
                No Hardware Changes
              </h3>
              <p className="text-[var(--text-secondary)]">
                Works with your existing CCTV infrastructure, no camera replacement needed
              </p>
          </div>
        </div>
      </div>
    </div>
  );
}
