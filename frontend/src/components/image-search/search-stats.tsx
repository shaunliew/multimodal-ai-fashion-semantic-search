/**
 * @fileoverview Search statistics dashboard component
 * @learning Displays system statistics and data distribution from the API
 * @concepts Data visualization, performance metrics, system monitoring
 */

'use client'

import { Database, Cpu, Users, Layers } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { cn } from '@/lib/utils'
import type { StatsResponse } from '@/types/api'

interface SearchStatsProps {
  stats: StatsResponse
  className?: string
}

/**
 * Component to display system statistics and data insights
 * @learning Shows API integration for monitoring and analytics
 */
export function SearchStats({ stats, className }: SearchStatsProps) {
  const embeddingCoverage = stats.database_stats.embedding_coverage

  return (
    <div className={cn('grid gap-4 md:grid-cols-2 lg:grid-cols-4', className)}>
      {/* Database Stats Card */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Database Status</CardTitle>
          <Database className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">
            {stats.database_stats.total_products.toLocaleString()}
          </div>
          <p className="text-xs text-muted-foreground">Total products</p>
          <div className="mt-3 space-y-1">
            <div className="flex items-center justify-between text-xs">
              <span>Embedding Coverage</span>
              <span className="font-medium">{embeddingCoverage}%</span>
            </div>
            <Progress value={embeddingCoverage} className="h-1.5" />
          </div>
        </CardContent>
      </Card>

      {/* System Info Card */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">AI Model</CardTitle>
          <Cpu className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-lg font-bold">CLIP</div>
          <p className="text-xs text-muted-foreground">
            {stats.system_info.embedding_dimensions}D embeddings
          </p>
          <div className="mt-3 space-y-1 text-xs">
            <div className="flex items-center justify-between">
              <span>Device</span>
              <span className="font-medium">{stats.system_info.device}</span>
            </div>
            <div className="flex items-center justify-between">
              <span>Similarity</span>
              <span className="font-medium">{stats.system_info.similarity_metric}</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Top Categories Card */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Top Categories</CardTitle>
          <Layers className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {stats.data_distribution.top_categories.slice(0, 3).map((cat) => (
              <div key={cat.category} className="flex items-center justify-between text-xs">
                <span className="truncate">{cat.category}</span>
                <span className="font-medium">{cat.count.toLocaleString()}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Gender Distribution Card */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Gender Distribution</CardTitle>
          <Users className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {stats.data_distribution.gender_distribution.map((gender) => {
              const percentage = Math.round(
                (gender.count / stats.database_stats.total_products) * 100
              )
              return (
                <div key={gender.gender} className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span>{gender.gender}</span>
                    <span className="font-medium">{percentage}%</span>
                  </div>
                  <Progress value={percentage} className="h-1.5" />
                </div>
              )
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  )
} 