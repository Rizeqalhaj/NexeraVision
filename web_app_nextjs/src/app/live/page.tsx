'use client';

import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Upload, Video, Grid3x3 } from 'lucide-react';
import { FileUpload } from './components/FileUpload';
import { LiveCamera } from './components/LiveCamera';
import { MultiCameraGrid } from './components/MultiCameraGrid';

export default function LiveDashboard() {
  return (
    <div className="container mx-auto p-6 max-w-7xl">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-[var(--text-primary)] mb-2">
          Live Detection
        </h1>
        <p className="text-[var(--text-secondary)]">
          Real-time violence detection tools for your CCTV system
        </p>
      </div>

      {/* Tabbed Interface */}
      <Tabs defaultValue="upload" className="w-full">
        <TabsList className="grid w-full grid-cols-3 mb-8 bg-gray-800/50 p-1 h-auto">
          <TabsTrigger
            value="upload"
            className="data-[state=active]:bg-[var(--accent-blue)] data-[state=active]:text-white py-3 px-4"
          >
            <Upload className="h-4 w-4 mr-2" />
            <span className="hidden sm:inline">File Upload</span>
            <span className="sm:hidden">Upload</span>
          </TabsTrigger>
          <TabsTrigger
            value="live"
            className="data-[state=active]:bg-[var(--danger-red)] data-[state=active]:text-white py-3 px-4"
          >
            <Video className="h-4 w-4 mr-2" />
            <span className="hidden sm:inline">Live Camera</span>
            <span className="sm:hidden">Live</span>
          </TabsTrigger>
          <TabsTrigger
            value="multi"
            className="data-[state=active]:bg-[var(--success-green)] data-[state=active]:text-white py-3 px-4"
          >
            <Grid3x3 className="h-4 w-4 mr-2" />
            <span className="hidden sm:inline">Multi-Camera Grid</span>
            <span className="sm:hidden">Grid</span>
          </TabsTrigger>
        </TabsList>

        {/* File Upload Tab */}
        <TabsContent value="upload" className="space-y-4">
          <div className="mb-4">
            <h2 className="text-2xl font-semibold text-[var(--text-primary)] mb-2">
              Upload Video for Detection
            </h2>
            <p className="text-[var(--text-secondary)]">
              Upload your CCTV footage to analyze violence probability using AI
            </p>
          </div>
          <FileUpload />
        </TabsContent>

        {/* Live Camera Tab */}
        <TabsContent value="live" className="space-y-4">
          <div className="mb-4">
            <h2 className="text-2xl font-semibold text-[var(--text-primary)] mb-2">
              Live Violence Detection
            </h2>
            <p className="text-[var(--text-secondary)]">
              Monitor your webcam feed in real-time with AI violence detection
            </p>
          </div>
          <LiveCamera />
        </TabsContent>

        {/* Multi-Camera Grid Tab */}
        <TabsContent value="multi" className="space-y-4">
          <div className="mb-4">
            <h2 className="text-2xl font-semibold text-[var(--text-primary)] mb-2">
              Multi-Camera Grid Monitoring
            </h2>
            <p className="text-[var(--text-secondary)]">
              Monitor multiple CCTV cameras simultaneously with screen recording segmentation
            </p>
          </div>
          <MultiCameraGrid />
        </TabsContent>
      </Tabs>

    </div>
  );
}
