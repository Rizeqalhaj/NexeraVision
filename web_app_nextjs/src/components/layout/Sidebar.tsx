'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  LayoutDashboard,
  Video,
  BarChart3,
  Camera,
  Bell,
  Settings,
  ChevronLeft,
  ChevronRight,
  LogOut
} from 'lucide-react';
import { useState } from 'react';
import { cn } from '@/lib/utils';

const menuItems = [
  {
    title: 'Dashboard',
    href: '/',
    icon: LayoutDashboard,
  },
  {
    title: 'Live Detection',
    href: '/live',
    icon: Video,
  },
  {
    title: 'Analysis',
    href: '/analysis',
    icon: BarChart3,
  },
  {
    title: 'Cameras',
    href: '/cameras',
    icon: Camera,
  },
  {
    title: 'Alerts',
    href: '/alerts',
    icon: Bell,
  },
  {
    title: 'Settings',
    href: '/settings',
    icon: Settings,
  },
];

export function Sidebar() {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className={cn(
      "relative h-screen bg-[var(--card-bg)] border-r border-[var(--border)] transition-all duration-300",
      collapsed ? "w-16" : "w-64"
    )}>
      {/* Logo */}
      <div className="h-16 flex items-center justify-between px-4 border-b border-[var(--border)]">
        {!collapsed && (
          <h1 className="text-xl font-bold text-[var(--text-primary)]">
            NexaraVision
          </h1>
        )}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="p-2 rounded-lg hover:bg-[var(--border)] transition-colors"
        >
          {collapsed ? (
            <ChevronRight className="h-4 w-4 text-[var(--text-secondary)]" />
          ) : (
            <ChevronLeft className="h-4 w-4 text-[var(--text-secondary)]" />
          )}
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto p-4 space-y-2">
        {menuItems.map((item) => {
          const Icon = item.icon;
          const isActive = pathname === item.href ||
                          (item.href !== '/' && pathname.startsWith(item.href));

          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-3 px-3 py-2 rounded-lg transition-all",
                "hover:bg-[var(--border)]",
                isActive && "bg-[var(--accent-blue)] text-white hover:bg-blue-600",
                !isActive && "text-[var(--text-secondary)]"
              )}
              title={collapsed ? item.title : undefined}
            >
              <Icon className="h-5 w-5 flex-shrink-0" />
              {!collapsed && (
                <span className="font-medium">{item.title}</span>
              )}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-[var(--border)] space-y-3">
        {/* Logout Button */}
        <button
          onClick={() => {
            // Handle logout logic here
            console.log('Logout clicked');
          }}
          className={cn(
            "flex items-center gap-3 px-3 py-2 rounded-lg transition-all w-full",
            "hover:bg-red-500/10 text-red-500 hover:text-red-600"
          )}
          title={collapsed ? 'Logout' : undefined}
        >
          <LogOut className="h-5 w-5 flex-shrink-0" />
          {!collapsed && (
            <span className="font-medium">Logout</span>
          )}
        </button>

        {/* Copyright */}
        {!collapsed && (
          <div className="text-xs text-[var(--text-secondary)]">
            <p>© 2025 NexaraVision</p>
            <p className="mt-1">Violence Detection AI</p>
          </div>
        )}
      </div>
    </div>
  );
}
