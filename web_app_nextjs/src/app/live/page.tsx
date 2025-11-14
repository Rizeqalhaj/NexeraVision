import Link from 'next/link';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Upload, Video, Grid3x3, ArrowRight } from 'lucide-react';

export default function LiveDashboard() {
  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <div className="container mx-auto px-6 py-16">
        <div className="text-center max-w-4xl mx-auto mb-16">
          <h1 className="text-6xl font-bold text-[var(--text-primary)] mb-6">
            NexaraVision
          </h1>
          <p className="text-2xl text-[var(--accent-blue)] mb-4">
            AI-Powered Violence Detection
          </p>
          <p className="text-lg text-[var(--text-secondary)] max-w-2xl mx-auto">
            Enterprise-grade violence detection for your existing CCTV infrastructure at 1/10th the cost.
            90-95% accuracy with real-time alerts.
          </p>
        </div>

        {/* Feature Cards */}
        <div className="grid md:grid-cols-3 gap-6 max-w-6xl mx-auto">
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
        <div className="mt-24 max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold text-[var(--text-primary)] text-center mb-12">
            Why NexaraVision?
          </h2>
          <div className="grid md:grid-cols-2 gap-8">
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
    </div>
  );
}
