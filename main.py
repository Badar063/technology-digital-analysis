
## Step 2: Technology Analysis Script

**analyze_tech.py**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class TechnologyDigitalAnalyzer:
    def __init__(self):
        """Initialize the analyzer and load datasets"""
        print("Loading technology and digital datasets...")
        self.website_data = pd.read_csv('data/website_traffic.csv')
        self.social_data = pd.read_csv('data/social_media_performance.csv')
        self.app_data = pd.read_csv('data/app_usage.csv')
        self.cohort_data = pd.read_csv('data/app_cohort_retention.csv')
        self.support_data = pd.read_csv('data/customer_support.csv')
        self.survey_data = pd.read_csv('data/tech_survey.csv')
        
        # Convert date columns to datetime
        self.website_data['date'] = pd.to_datetime(self.website_data['date'])
        self.social_data['post_date'] = pd.to_datetime(self.social_data['post_date'])
        self.app_data['date'] = pd.to_datetime(self.app_data['date'])
        self.support_data['date'] = pd.to_datetime(self.support_data['date'])
        
        print("Data loaded successfully!")
    
    def analyze_website_traffic(self):
        """Analyze website traffic patterns and user engagement metrics"""
        print("\n" + "="*60)
        print("ANALYSIS 1: WEBSITE TRAFFIC PATTERNS AND USER ENGAGEMENT")
        print("="*60)
        
        # Calculate weekly aggregates
        self.website_data['week'] = self.website_data['date'].dt.isocalendar().week
        self.website_data['week_start'] = self.website_data['date'] - pd.to_timedelta(self.website_data['date'].dt.dayofweek, unit='D')
        
        weekly_metrics = self.website_data.groupby('week_start').agg({
            'visitors': 'sum',
            'pageviews': 'sum',
            'bounce_rate': 'mean',
            'avg_session_duration_seconds': 'mean',
            'conversion_rate': 'mean',
            'revenue': 'sum'
        }).reset_index()
        
        # Calculate growth rates
        weekly_metrics['visitor_growth'] = weekly_metrics['visitors'].pct_change() * 100
        weekly_metrics['revenue_growth'] = weekly_metrics['revenue'].pct_change() * 100
        
        # Traffic source analysis
        traffic_sources = ['organic_search_traffic', 'direct_traffic', 'social_media_traffic',
                          'referral_traffic', 'paid_search_traffic']
        
        total_traffic = self.website_data[traffic_sources].sum()
        traffic_percentages = (total_traffic / total_traffic.sum() * 100).round(2)
        
        # Device analysis
        device_columns = ['mobile_percent', 'desktop_percent', 'tablet_percent']
        device_avg = self.website_data[device_columns].mean().round(2)
        
        print("\nTraffic Source Distribution:")
        for source, percent in traffic_percentages.items():
            print(f"  {source.replace('_traffic', '').replace('_', ' ').title()}: {percent}%")
        
        print("\nDevice Usage Distribution:")
        for device, percent in device_avg.items():
            print(f"  {device.replace('_percent', '').title()}: {percent}%")
        
        print(f"\nAverage Weekly Metrics:")
        print(f"  Visitors: {weekly_metrics['visitors'].mean():.0f}")
        print(f"  Bounce Rate: {weekly_metrics['bounce_rate'].mean():.1f}%")
        print(f"  Session Duration: {weekly_metrics['avg_session_duration_seconds'].mean():.0f} seconds")
        print(f"  Conversion Rate: {weekly_metrics['conversion_rate'].mean():.2f}%")
        print(f"  Revenue: ${weekly_metrics['revenue'].mean():.2f}")
        
        # Correlation analysis
        correlation_matrix = self.website_data[['visitors', 'bounce_rate', 
                                               'avg_session_duration_seconds', 
                                               'pages_per_session', 'conversion_rate',
                                               'revenue']].corr()
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Website Traffic Analysis', fontsize=16, fontweight='bold')
        
        # 1. Traffic trends over time
        axes[0, 0].plot(self.website_data['date'], self.website_data['visitors'], 
                       label='Visitors', alpha=0.8, linewidth=2)
        axes[0, 0].plot(self.website_data['date'], self.website_data['pageviews'] / 10, 
                       label='Pageviews/10', alpha=0.6, linewidth=1.5)
        axes[0, 0].set_title('Traffic Volume Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Traffic sources pie chart
        source_labels = [s.replace('_traffic', '').replace('_', ' ').title() 
                        for s in traffic_percentages.index]
        colors = plt.cm.Set3(np.linspace(0, 1, len(traffic_percentages)))
        axes[0, 1].pie(traffic_percentages.values, labels=source_labels, 
                      autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0, 1].set_title('Traffic Source Distribution')
        
        # 3. Day of week patterns
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_patterns = self.website_data.groupby('day_of_week').agg({
            'visitors': 'mean',
            'conversion_rate': 'mean',
            'bounce_rate': 'mean'
        }).reindex(day_order)
        
        x = np.arange(len(day_order))
        width = 0.25
        
        axes[1, 0].bar(x - width, day_patterns['visitors'], width, 
                      label='Avg Visitors', alpha=0.7)
        axes[1, 0].bar(x, day_patterns['conversion_rate'], width, 
                      label='Conversion Rate', alpha=0.7)
        
        ax2 = axes[1, 0].twinx()
        ax2.plot(x, day_patterns['bounce_rate'], color='red', 
                marker='o', linewidth=2, label='Bounce Rate')
        
        axes[1, 0].set_title('Day of Week Patterns')
        axes[1, 0].set_xlabel('Day of Week')
        axes[1, 0].set_ylabel('Visitors / Conversion Rate', color='blue')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(day_order, rotation=45)
        axes[1, 0].tick_params(axis='y', labelcolor='blue')
        axes[1, 0].legend(loc='upper left')
        
        ax2.set_ylabel('Bounce Rate (%)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.legend(loc='upper right')
        
        # 4. Correlation heatmap
        im = axes[1, 1].imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        axes[1, 1].set_title('Metrics Correlation Matrix')
        axes[1, 1].set_xticks(range(len(correlation_matrix.columns)))
        axes[1, 1].set_yticks(range(len(correlation_matrix.columns)))
        axes[1, 1].set_xticklabels([col[:15] for col in correlation_matrix.columns], rotation=45, ha='right')
        axes[1, 1].set_yticklabels([col[:15] for col in correlation_matrix.columns])
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                axes[1, 1].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                               ha='center', va='center', color='black', fontsize=8)
        
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig('website_traffic_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return weekly_metrics, traffic_percentages
    
    def analyze_social_media(self):
        """Study social media post performance and audience interaction"""
        print("\n" + "="*60)
        print("ANALYSIS 2: SOCIAL MEDIA PERFORMANCE AND AUDIENCE INTERACTION")
        print("="*60)
        
        # Platform comparison
        platform_stats = self.social_data.groupby('platform').agg({
            'impressions': 'mean',
            'engagement_rate': 'mean',
            'engagements': 'mean',
            'virality_score': 'mean'
        }).round(2)
        
        print("\nPlatform Performance Comparison:")
        print(platform_stats)
        
        # Content type analysis
        content_stats = self.social_data.groupby('post_type').agg({
            'engagement_rate': 'mean',
            'impressions': 'mean',
            'virality_score': 'mean'
        }).round(2)
        
        print("\nContent Type Performance:")
        print(content_stats.sort_values('engagement_rate', ascending=False))
        
        # Best performing posts
        top_posts = self.social_data.nlargest(5, 'engagement_rate')
        print("\nTop 5 Performing Posts:")
        for _, post in top_posts.iterrows():
            print(f"  {post['platform']} - {post['post_type']}: "
                  f"Engagement Rate: {post['engagement_rate']}%, "
                  f"Impressions: {post['impressions']:,}")
        
        # Time of day analysis
        self.social_data['time_of_day'] = pd.cut(self.social_data['post_hour'], 
                                                 bins=[0, 6, 12, 18, 24],
                                                 labels=['Night', 'Morning', 'Afternoon', 'Evening'])
        
        time_stats = self.social_data.groupby('time_of_day').agg({
            'engagement_rate': 'mean',
            'impressions': 'mean'
        }).round(2)
        
        print("\nPerformance by Time of Day:")
        print(time_stats)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Social Media Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Platform comparison
        platforms = platform_stats.index
        x = np.arange(len(platforms))
        width = 0.2
        
        metrics = ['impressions', 'engagement_rate', 'engagements', 'virality_score']
        metric_labels = ['Impressions', 'Engagement Rate', 'Engagements', 'Virality Score']
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            offset = width * (i - 1.5)
            if metric == 'impressions':
                axes[0, 0].bar(x + offset, platform_stats[metric] / 1000, width, 
                              label=f'{label} (K)', alpha=0.7)
            else:
                axes[0, 0].bar(x + offset, platform_stats[metric], width, 
                              label=label, alpha=0.7)
        
        axes[0, 0].set_title('Platform Performance Comparison')
        axes[0, 0].set_xlabel('Platform')
        axes[0, 0].set_ylabel('Metric Value')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(platforms)
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 2. Content type performance
        content_types = content_stats.index
        x_pos = np.arange(len(content_types))
        
        ax1 = axes[0, 1]
        ax2 = ax1.twinx()
        
        bars = ax1.bar(x_pos, content_stats['engagement_rate'], 
                      alpha=0.7, color='blue', label='Engagement Rate')
        ax1.set_xlabel('Content Type')
        ax1.set_ylabel('Engagement Rate (%)', color='blue')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(content_types, rotation=45)
        ax1.tick_params(axis='y', labelcolor='blue')
        
        ax2.plot(x_pos, content_stats['impressions'] / 1000, 
                color='red', marker='o', linewidth=2, label='Impressions (K)')
        ax2.set_ylabel('Impressions (Thousands)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        axes[0, 1].set_title('Content Type Performance')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 3. Time of day performance
        time_stats_sorted = time_stats.reindex(['Morning', 'Afternoon', 'Evening', 'Night'])
        
        x_time = np.arange(len(time_stats_sorted))
        width_time = 0.35
        
        axes[1, 0].bar(x_time - width_time/2, time_stats_sorted['engagement_rate'], 
                      width_time, label='Engagement Rate', alpha=0.7)
        axes[1, 0].bar(x_time + width_time/2, time_stats_sorted['impressions'] / 1000, 
                      width_time, label='Impressions (K)', alpha=0.7)
        
        axes[1, 0].set_title('Performance by Time of Day')
        axes[1, 0].set_xlabel('Time of Day')
        axes[1, 0].set_ylabel('Metric Value')
        axes[1, 0].set_xticks(x_time)
        axes[1, 0].set_xticklabels(time_stats_sorted.index)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Engagement vs impressions scatter
        scatter = axes[1, 1].scatter(self.social_data['impressions'] / 1000,
                                    self.social_data['engagement_rate'],
                                    c=self.social_data['virality_score'],
                                    cmap='viridis', s=50, alpha=0.7)
        
        # Add platform labels
        platforms_unique = self.social_data['platform'].unique()
        for platform in platforms_unique:
            platform_data = self.social_data[self.social_data['platform'] == platform]
            axes[1, 1].scatter([], [], label=platform, alpha=0.7)
        
        axes[1, 1].set_title('Impressions vs Engagement Rate')
        axes[1, 1].set_xlabel('Impressions (Thousands)')
        axes[1, 1].set_ylabel('Engagement Rate (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.colorbar(scatter, ax=axes[1, 1], label='Virality Score')
        
        plt.tight_layout()
        plt.savefig('social_media_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return platform_stats, content_stats
    
    def analyze_app_usage(self):
        """Examine app usage data and user retention rates"""
        print("\n" + "="*60)
        print("ANALYSIS 3: APP USAGE DATA AND USER RETENTION RATES")
        print("="*60)
        
        # Calculate key metrics
        avg_dau = self.app_data['daily_active_users'].mean()
        avg_mau = self.app_data['monthly_active_users'].mean()
        stickiness = (avg_dau / avg_mau * 100) if avg_mau > 0 else 0
        
        avg_session_length = self.app_data['avg_session_length_seconds'].mean()
        avg_retention = self.app_data['retention_rate_day'].mean()
        
        print(f"\nKey App Metrics:")
        print(f"  Average DAU: {avg_dau:.0f}")
        print(f"  Average MAU: {avg_mau:.0f}")
        print(f"  Stickiness (DAU/MAU): {stickiness:.2f}%")
        print(f"  Average Session Length: {avg_session_length:.0f} seconds")
        print(f"  Average Day 1 Retention: {avg_retention:.2f}%")
        
        # Feature usage analysis
        feature_cols = ['messaging_usage_percent', 'profile_view_usage_percent',
                       'search_usage_percent', 'notification_usage_percent',
                       'settings_usage_percent']
        
        feature_usage = self.app_data[feature_cols].mean().round(2)
        
        print("\nFeature Usage (% of DAU):")
        for feature, usage in feature_usage.items():
            feature_name = feature.replace('_usage_percent', '').replace('_', ' ').title()
            print(f"  {feature_name}: {usage}%")
        
        # Platform comparison
        ios_share = (self.app_data['ios_users'].sum() / 
                    (self.app_data['ios_users'].sum() + self.app_data['android_users'].sum()) * 100)
        
        print(f"\nPlatform Distribution:")
        print(f"  iOS: {ios_share:.1f}%")
        print(f"  Android: {100 - ios_share:.1f}%")
        
        # Cohort retention analysis
        cohort_summary = self.cohort_data.groupby('day_number')['retention_percent'].agg(['mean', 'std']).reset_index()
        
        print("\nCohort Retention Analysis:")
        for day in [1, 7, 14, 30]:
            if day <= len(cohort_summary):
                retention = cohort_summary.loc[cohort_summary['day_number'] == day, 'mean'].values[0]
                print(f"  Day {day}: {retention:.1f}% retention")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('App Usage Analysis', fontsize=16, fontweight='bold')
        
        # 1. Daily metrics trend
        axes[0, 0].plot(self.app_data['date'], self.app_data['daily_active_users'], 
                       label='DAU', linewidth=2, alpha=0.8)
        axes[0, 0].plot(self.app_data['date'], self.app_data['total_sessions'] / 10, 
                       label='Sessions/10', linewidth=1.5, alpha=0.6)
        axes[0, 0].set_title('Daily Active Users and Sessions')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Feature usage comparison
        features_sorted = feature_usage.sort_values(ascending=False)
        colors = plt.cm.Set3(np.linspace(0, 1, len(features_sorted)))
        axes[0, 1].barh(range(len(features_sorted)), features_sorted.values, 
                       color=colors, alpha=0.7)
        axes[0, 1].set_yticks(range(len(features_sorted)))
        axes[0, 1].set_yticklabels([f.replace('_usage_percent', '').replace('_', ' ').title() 
                                   for f in features_sorted.index])
        axes[0, 1].set_title('Feature Usage (% of DAU)')
        axes[0, 1].set_xlabel('Usage Percentage')
        axes[0, 1].grid(True, alpha=0.3, axis='x')
        
        # 3. Cohort retention curves
        cohort_dates = self.cohort_data['cohort_date'].unique()[:5]  # Show first 5 cohorts
        
        for cohort_date in cohort_dates:
            cohort_subset = self.cohort_data[self.cohort_data['cohort_date'] == cohort_date]
            axes[1, 0].plot(cohort_subset['day_number'], cohort_subset['retention_percent'],
                           label=cohort_date, marker='.', alpha=0.7)
        
        axes[1, 0].set_title('Cohort Retention Curves')
        axes[1, 0].set_xlabel('Days Since Signup')
        axes[1, 0].set_ylabel('Retention Rate (%)')
        axes[1, 0].legend(title='Cohort Date')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Correlation matrix for app metrics
        correlation_cols = ['daily_active_users', 'stickiness_percent', 
                          'avg_session_length_seconds', 'retention_rate_day',
                          'engagement_score', 'conversion_rate', 'app_store_rating']
        
        app_corr = self.app_data[correlation_cols].corr()
        
        im = axes[1, 1].imshow(app_corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        axes[1, 1].set_title('App Metrics Correlation Matrix')
        axes[1, 1].set_xticks(range(len(correlation_cols)))
        axes[1, 1].set_yticks(range(len(correlation_cols)))
        axes[1, 1].set_xticklabels([col[:15] for col in correlation_cols], rotation=45, ha='right')
        axes[1, 1].set_yticklabels([col[:15] for col in correlation_cols])
        
        for i in range(len(correlation_cols)):
            for j in range(len(correlation_cols)):
                axes[1, 1].text(j, i, f'{app_corr.iloc[i, j]:.2f}',
                               ha='center', va='center', color='black', fontsize=8)
        
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig('app_usage_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return feature_usage, cohort_summary
    
    def analyze_customer_support(self):
        """Track customer support ticket volumes and resolution times"""
        print("\n" + "="*60)
        print("ANALYSIS 4: CUSTOMER SUPPORT TICKETS AND RESOLUTION TIMES")
        print("="*60)
        
        # Calculate weekly aggregates
        self.support_data['week'] = self.support_data['date'].dt.isocalendar().week
        weekly_summary = self.support_data.groupby('week').agg({
            'tickets_received': 'sum',
            'tickets_resolved': 'sum',
            'avg_resolution_time_hours': 'mean',
            'csat_score': 'mean',
            'backlog': 'last'
        }).reset_index()
        
        # Efficiency metrics
        resolution_rate = (self.support_data['tickets_resolved'].sum() / 
                          self.support_data['tickets_received'].sum() * 100)
        
        avg_resolution_time = self.support_data['avg_resolution_time_hours'].mean()
        avg_csat = self.support_data['csat_score'].mean()
        
        print(f"\nSupport Efficiency Metrics:")
        print(f"  Total Tickets Received: {self.support_data['tickets_received'].sum():,}")
        print(f"  Total Tickets Resolved: {self.support_data['tickets_resolved'].sum():,}")
        print(f"  Resolution Rate: {resolution_rate:.1f}%")
        print(f"  Average Resolution Time: {avg_resolution_time:.1f} hours")
        print(f"  Average CSAT Score: {avg_csat:.1f}/5.0")
        print(f"  Final Backlog: {self.support_data['backlog'].iloc[-1]:,} tickets")
        
        # Category analysis
        category_cols = ['technical_tickets', 'billing_tickets', 'account_tickets',
                        'feature_request_tickets', 'general_tickets']
        
        category_totals = self.support_data[category_cols].sum()
        category_percentages = (category_totals / category_totals.sum() * 100).round(2)
        
        print("\nTicket Category Distribution:")
        for category, percent in category_percentages.items():
            category_name = category.replace('_tickets', '').replace('_', ' ').title()
            print(f"  {category_name}: {percent}%")
        
        # Channel analysis
        channel_cols = ['email_tickets', 'chat_tickets', 'phone_tickets',
                       'social_media_tickets', 'self_service_tickets']
        
        channel_totals = self.support_data[channel_cols].sum()
        
        print("\nTicket Channel Distribution:")
        for channel, total in channel_totals.items():
            channel_name = channel.replace('_tickets', '').replace('_', ' ').title()
            percentage = (total / channel_totals.sum() * 100)
            print(f"  {channel_name}: {total:,} ({percentage:.1f}%)")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Customer Support Analysis', fontsize=16, fontweight='bold')
        
        # 1. Ticket volume trends
        axes[0, 0].plot(self.support_data['date'], self.support_data['tickets_received'],
                       label='Received', linewidth=2, alpha=0.8)
        axes[0, 0].plot(self.support_data['date'], self.support_data['tickets_resolved'],
                       label='Resolved', linewidth=2, alpha=0.8)
        axes[0, 0].fill_between(self.support_data['date'], 0, 
                               self.support_data['backlog'] / 10,
                               color='red', alpha=0.2, label='Backlog/10')
        axes[0, 0].set_title('Ticket Volume and Backlog')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Ticket Count')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Category distribution
        categories_sorted = category_percentages.sort_values(ascending=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories_sorted)))
        axes[0, 1].barh(range(len(categories_sorted)), categories_sorted.values,
                       color=colors, alpha=0.7)
        axes[0, 1].set_yticks(range(len(categories_sorted)))
        axes[0, 1].set_yticklabels([c.replace('_tickets', '').replace('_', ' ').title() 
                                   for c in categories_sorted.index])
        axes[0, 1].set_title('Ticket Category Distribution')
        axes[0, 1].set_xlabel('Percentage of Tickets')
        axes[0, 1].grid(True, alpha=0.3, axis='x')
        
        # 3. Resolution time vs CSAT scatter
        scatter = axes[1, 0].scatter(self.support_data['avg_resolution_time_hours'],
                                    self.support_data['csat_score'],
                                    c=self.support_data['escalation_rate'],
                                    cmap='RdYlGn_r', s=50, alpha=0.7)
        
        # Add trend line
        z = np.polyfit(self.support_data['avg_resolution_time_hours'],
                      self.support_data['csat_score'], 1)
        p = np.poly1d(z)
        axes[1, 0].plot(self.support_data['avg_resolution_time_hours'],
                       p(self.support_data['avg_resolution_time_hours']),
                       "r--", alpha=0.8, 
                       label=f'Correlation: {np.corrcoef(self.support_data["avg_resolution_time_hours"], self.support_data["csat_score"])[0,1]:.3f}')
        
        axes[1, 0].set_title('Resolution Time vs Customer Satisfaction')
        axes[1, 0].set_xlabel('Average Resolution Time (hours)')
        axes[1, 0].set_ylabel('CSAT Score (1-5)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        plt.colorbar(scatter, ax=axes[1, 0], label='Escalation Rate (%)')
        
        # 4. Day of week patterns
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_patterns = self.support_data.groupby('day_of_week').agg({
            'tickets_received': 'mean',
            'avg_resolution_time_hours': 'mean',
            'csat_score': 'mean'
        }).reindex(day_order)
        
        x = np.arange(len(day_order))
        width = 0.25
        
        axes[1, 1].bar(x - width, day_patterns['tickets_received'], width,
                      label='Avg Tickets', alpha=0.7)
        axes[1, 1].bar(x, day_patterns['avg_resolution_time_hours'], width,
                      label='Resolution Time (hrs)', alpha=0.7)
        
        ax2 = axes[1, 1].twinx()
        ax2.plot(x, day_patterns['csat_score'], color='red',
                marker='o', linewidth=2, label='CSAT Score')
        
        axes[1, 1].set_title('Day of Week Patterns')
        axes[1, 1].set_xlabel('Day of Week')
        axes[1, 1].set_ylabel('Tickets / Resolution Time', color='blue')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(day_order, rotation=45)
        axes[1, 1].tick_params(axis='y', labelcolor='blue')
        axes[1, 1].legend(loc='upper left')
        
        ax2.set_ylabel('CSAT Score (1-5)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig('customer_support_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return weekly_summary, category_percentages
    
    def analyze_tech_survey(self):
        """Analyze survey responses about technology adoption and preferences"""
        print("\n" + "="*60)
        print("ANALYSIS 5: TECHNOLOGY ADOPTION SURVEY RESPONSES")
        print("="*60)
        
        # Demographic summary
        print("\nDemographic Summary:")
        print(f"  Average Age: {self.survey_data['age'].mean():.1f} years")
        print(f"  Gender Distribution:")
        for gender, count in self.survey_data['gender'].value_counts().items():
            percentage = (count / len(self.survey_data)) * 100
            print(f"    {gender}: {count} ({percentage:.1f}%)")
        
        # Tech proficiency distribution
        print("\nTech Proficiency Distribution:")
        for proficiency, count in self.survey_data['tech_proficiency'].value_counts().items():
            percentage = (count / len(self.survey_data)) * 100
            print(f"  {proficiency}: {count} ({percentage:.1f}%)")
        
        # Digital literacy scores by age group
        self.survey_data['age_group'] = pd.cut(self.survey_data['age'], 
                                              bins=[18, 25, 35, 50, 65, 100],
                                              labels=['18-25', '26-35', '36-50', '51-65', '65+'])
        
        literacy_by_age = self.survey_data.groupby('age_group').agg({
            'digital_literacy_score': 'mean',
            'tech_adoption_score': 'mean',
            'tech_anxiety_score': 'mean'
        }).round(2)
        
        print("\nDigital Literacy by Age Group:")
        print(literacy_by_age)
        
        # AI familiarity analysis
        ai_familiarity_counts = self.survey_data['ai_familiarity'].value_counts()
        print("\nAI Familiarity Levels:")
        for level, count in ai_familiarity_counts.items():
            percentage = (count / len(self.survey_data)) * 100
            print(f"  {level}: {count} ({percentage:.1f}%)")
        
        # Correlation analysis
        correlation_cols = ['age', 'tech_adoption_score', 'digital_literacy_score',
                          'tech_anxiety_score', 'tech_productivity_impact',
                          'security_concern_level', 'privacy_concern_level']
        
        survey_corr = self.survey_data[correlation_cols].corr()
        
        print("\nKey Correlations:")
        print(f"  Age vs Digital Literacy: {survey_corr.loc['age', 'digital_literacy_score']:.3f}")
        print(f"  Digital Literacy vs Tech Anxiety: {survey_corr.loc['digital_literacy_score', 'tech_anxiety_score']:.3f}")
        print(f"  Tech Adoption vs Productivity: {survey_corr.loc['tech_adoption_score', 'tech_productivity_impact']:.3f}")
        
        # Cluster analysis for tech adoption segments
        cluster_cols = ['age', 'tech_adoption_score', 'digital_literacy_score',
                       'tech_anxiety_score', 'daily_screen_time_hours',
                       'social_media_hours_daily']
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.survey_data[cluster_cols])
        
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        self.survey_data['tech_segment'] = kmeans.fit_predict(scaled_data)
        
        segment_profiles = self.survey_data.groupby('tech_segment').agg({
            'age': 'mean',
            'tech_adoption_score': 'mean',
            'digital_literacy_score': 'mean',
            'tech_anxiety_score': 'mean',
            'tech_proficiency': lambda x: x.mode()[0],
            'primary_device': lambda x: x.mode()[0]
        }).round(2)
        
        print("\nTechnology Adoption Segments:")
        print(segment_profiles)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Technology Adoption Survey Analysis', fontsize=16, fontweight='bold')
        
        # 1. Age vs digital literacy scatter with segments
        scatter = axes[0, 0].scatter(self.survey_data['age'],
                                    self.survey_data['digital_literacy_score'],
                                    c=self.survey_data['tech_segment'],
                                    cmap='viridis', s=50, alpha=0.7)
        
        # Add trend line
        z = np.polyfit(self.survey_data['age'], 
                      self.survey_data['digital_literacy_score'], 1)
        p = np.poly1d(z)
        axes[0, 0].plot(self.survey_data['age'], 
                       p(self.survey_data['age']),
                       "r--", alpha=0.8, 
                       label=f'Correlation: {np.corrcoef(self.survey_data["age"], self.survey_data["digital_literacy_score"])[0,1]:.3f}')
        
        axes[0, 0].set_title('Age vs Digital Literacy Score')
        axes[0, 0].set_xlabel('Age')
        axes[0, 0].set_ylabel('Digital Literacy Score (1-10)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        plt.colorbar(scatter, ax=axes[0, 0], label='Tech Segment')
        
        # 2. Tech proficiency distribution
        proficiency_counts = self.survey_data['tech_proficiency'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(proficiency_counts)))
        axes[0, 1].pie(proficiency_counts.values, labels=proficiency_counts.index,
                      autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0, 1].set_title('Tech Proficiency Distribution')
        
        # 3. Correlation heatmap
        im = axes[1, 0].imshow(survey_corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        axes[1, 0].set_title('Survey Variables Correlation Matrix')
        axes[1, 0].set_xticks(range(len(survey_corr.columns)))
        axes[1, 0].set_yticks(range(len(survey_corr.columns)))
        axes[1, 0].set_xticklabels([col[:15] for col in survey_corr.columns], rotation=45, ha='right')
        axes[1, 0].set_yticklabels([col[:15] for col in survey_corr.columns])
        
        for i in range(len(survey_corr.columns)):
            for j in range(len(survey_corr.columns)):
                axes[1, 0].text(j, i, f'{survey_corr.iloc[i, j]:.2f}',
                               ha='center', va='center', color='black', fontsize=8)
        
        plt.colorbar(im, ax=axes[1, 0])
        
        # 4. Segment comparison radar chart (simplified to bar chart)
        segment_metrics = ['tech_adoption_score', 'digital_literacy_score',
                          'tech_productivity_impact', 'future_tech_interest']
        
        segment_comparison = pd.DataFrame()
        for segment in sorted(self.survey_data['tech_segment'].unique()):
            segment_data = self.survey_data[self.survey_data['tech_segment'] == segment]
            for metric in segment_metrics:
                segment_comparison.loc[segment, metric] = segment_data[metric].mean()
        
        # Normalize for radar-like comparison
        segment_normalized = segment_comparison.copy()
        for col in segment_metrics:
            segment_normalized[col] = segment_normalized[col] / segment_normalized[col].max()
        
        x = np.arange(len(segment_metrics))
        width = 0.2
        
        for i, segment in enumerate(segment_normalized.index):
            offset = width * (i - 1.5)
            axes[1, 1].bar(x + offset, segment_normalized.loc[segment].values,
                          width, label=f'Segment {segment}', alpha=0.7)
        
        axes[1, 1].set_title('Technology Adoption Segments Comparison')
        axes[1, 1].set_xlabel('Metric')
        axes[1, 1].set_ylabel('Normalized Score')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([m.replace('_score', '').replace('_', ' ').title() 
                                   for m in segment_metrics], rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('tech_survey_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return literacy_by_age, segment_profiles
    
    def run_all_analysis(self):
        """Run all analyses and generate comprehensive report"""
        print("\n" + "="*60)
        print("COMPREHENSIVE TECHNOLOGY & DIGITAL ANALYSIS")
        print("="*60)
        
        results = {}
        
        print("\n1. Analyzing Website Traffic...")
        results['website'] = self.analyze_website_traffic()
        
        print("\n2. Analyzing Social Media Performance...")
        results['social'] = self.analyze_social_media()
        
        print("\n3. Analyzing App Usage...")
        results['app'] = self.analyze_app_usage()
        
        print("\n4. Analyzing Customer Support...")
        results['support'] = self.analyze_customer_support()
        
        print("\n5. Analyzing Technology Survey...")
        results['survey'] = self.analyze_tech_survey()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        
        # Generate summary report
        self.generate_summary_report(results)
        
        return results
    
    def generate_summary_report(self, results):
        """Generate a summary report of all analyses"""
        report = []
        report.append("="*60)
        report.append("TECHNOLOGY & DIGITAL ANALYSIS SUMMARY REPORT")
        report.append("="*60)
        report.append("")
        
        # Website summary
        website_weekly = results['website'][0]
        report.append("1. WEBSITE TRAFFIC ANALYSIS")
        report.append("-"*40)
        report.append(f"Average Weekly Visitors: {website_weekly['visitors'].mean():.0f}")
        report.append(f"Average Conversion Rate: {website_weekly['conversion_rate'].mean():.2f}%")
        report.append(f"Average Bounce Rate: {website_weekly['bounce_rate'].mean():.1f}%")
        report.append("")
        
        # Social media summary
        platform_stats = results['social'][0]
        best_platform = platform_stats['engagement_rate'].idxmax()
        report.append("2. SOCIAL MEDIA ANALYSIS")
        report.append("-"*40)
        report.append(f"Best Performing Platform: {best_platform}")
        report.append(f"Average Engagement Rate: {platform_stats['engagement_rate'].mean():.2f}%")
        report.append("")
        
        # App usage summary
        feature_usage = results['app'][0]
        top_feature = feature_usage.idxmax()
        report.append("3. APP USAGE ANALYSIS")
        report.append("-"*40)
        report.append(f"Most Used Feature: {top_feature.replace('_usage_percent', '').replace('_', ' ').title()}")
        report.append(f"Usage Rate: {feature_usage.max():.1f}%")
        report.append("")
        
        # Customer support summary
        support_weekly = results['support'][0]
        report.append("4. CUSTOMER SUPPORT ANALYSIS")
        report.append("-"*40)
        report.append(f"Average Weekly Tickets: {support_weekly['tickets_received'].mean():.0f}")
        report.append(f"Average Resolution Time: {support_weekly['avg_resolution_time_hours'].mean():.1f} hours")
        report.append(f"Average CSAT Score: {support_weekly['csat_score'].mean():.1f}/5.0")
        report.append("")
        
        # Survey summary
        segment_profiles = results['survey'][1]
        report.append("5. TECHNOLOGY ADOPTION SURVEY")
        report.append("-"*40)
        report.append(f"Number of Tech Segments Identified: {len(segment_profiles)}")
        report.append("Segment Characteristics:")
        for segment, row in segment_profiles.iterrows():
            report.append(f"  Segment {segment}: Age={row['age']:.1f}, "
                         f"Adoption={row['tech_adoption_score']:.1f}, "
                         f"Literacy={row['digital_literacy_score']:.1f}")
        
        # Key insights and recommendations
        report.append("")
        report.append("KEY INSIGHTS & RECOMMENDATIONS")
        report.append("-"*40)
        report.append("1. Website: Focus on reducing bounce rate through better mobile optimization")
        report.append("2. Social Media: Invest more in video content and TikTok platform")
        report.append("3. App: Improve feature discovery for underutilized features")
        report.append("4. Support: Reduce resolution times to improve customer satisfaction")
        report.append("5. Technology: Develop targeted onboarding for different tech proficiency segments")
        
        # Save report to file
        with open('tech_analysis_summary.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print("\nSummary report saved to 'tech_analysis_summary.txt'")
        print("Visualizations saved as PNG files:")
        print("- website_traffic_analysis.png")
        print("- social_media_analysis.png")
        print("- app_usage_analysis.png")
        print("- customer_support_analysis.png")
        print("- tech_survey_analysis.png")

def main():
    """Main function to run the analysis"""
    print("Initializing Technology & Digital Analyzer...")
    analyzer = TechnologyDigitalAnalyzer()
    
    # Run all analyses
    results = analyzer.run_all_analysis()
    
    print("\n" + "="*60)
    print("All analyses completed successfully!")
    print("Check the generated files for detailed results.")
    print("="*60)

if __name__ == "__main__":
    main()
