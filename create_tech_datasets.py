import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_website_traffic_data():
    """Create website traffic patterns and user engagement metrics"""
    np.random.seed(57)
    
    # Generate 90 days of data
    dates = []
    start_date = datetime(2024, 1, 1)
    for i in range(90):
        date = start_date + timedelta(days=i)
        dates.append(date.strftime('%Y-%m-%d'))
    
    days_of_week = [(start_date + timedelta(days=i)).strftime('%A') for i in range(90)]
    
    data = []
    
    for i, (date, day) in enumerate(zip(dates, days_of_week)):
        # Base traffic with weekly patterns
        if day in ['Monday', 'Tuesday', 'Wednesday']:
            base_visitors = np.random.randint(5000, 8000)
        elif day in ['Thursday', 'Friday']:
            base_visitors = np.random.randint(4000, 7000)
        else:  # Weekend
            base_visitors = np.random.randint(2000, 5000)
        
        # Seasonality and growth trend
        seasonal_factor = 1 + 0.2 * np.sin(i * 2 * np.pi / 7)  # Weekly seasonality
        growth_factor = 1 + i * 0.002  # Slow growth over time
        
        visitors = int(base_visitors * seasonal_factor * growth_factor * np.random.uniform(0.9, 1.1))
        
        # Engagement metrics
        bounce_rate = np.random.uniform(35, 55)
        if day in ['Saturday', 'Sunday']:
            bounce_rate += np.random.uniform(5, 10)  # Higher bounce on weekends
        
        avg_session_duration = np.random.uniform(120, 300)  # seconds
        if visitors > 7000:
            avg_session_duration *= np.random.uniform(0.9, 1.0)  # Slightly lower for high traffic
        
        pages_per_session = np.random.uniform(3.5, 6.5)
        
        # Traffic sources
        organic_search = visitors * np.random.uniform(0.35, 0.45)
        direct_traffic = visitors * np.random.uniform(0.20, 0.30)
        social_media = visitors * np.random.uniform(0.10, 0.20)
        referral = visitors * np.random.uniform(0.08, 0.15)
        paid_search = visitors * np.random.uniform(0.05, 0.12)
        
        # Conversion metrics
        conversion_rate = np.random.uniform(1.5, 3.5)
        if day in ['Tuesday', 'Wednesday']:
            conversion_rate += np.random.uniform(0.5, 1.0)
        
        conversions = int(visitors * conversion_rate / 100)
        revenue = conversions * np.random.uniform(50, 200)
        
        # Device breakdown
        mobile_percent = np.random.uniform(55, 75)
        desktop_percent = np.random.uniform(25, 40)
        tablet_percent = 100 - mobile_percent - desktop_percent
        
        data.append({
            'date': date,
            'day_of_week': day,
            'visitors': visitors,
            'pageviews': int(visitors * pages_per_session),
            'unique_visitors': int(visitors * np.random.uniform(0.7, 0.9)),
            'bounce_rate': round(bounce_rate, 2),
            'avg_session_duration_seconds': round(avg_session_duration, 1),
            'pages_per_session': round(pages_per_session, 2),
            'organic_search_traffic': int(organic_search),
            'direct_traffic': int(direct_traffic),
            'social_media_traffic': int(social_media),
            'referral_traffic': int(referral),
            'paid_search_traffic': int(paid_search),
            'conversion_rate': round(conversion_rate, 2),
            'conversions': conversions,
            'revenue': round(revenue, 2),
            'mobile_percent': round(mobile_percent, 2),
            'desktop_percent': round(desktop_percent, 2),
            'tablet_percent': round(tablet_percent, 2),
            'new_visitors_percent': round(np.random.uniform(60, 80), 2)
        })
    
    return pd.DataFrame(data)

def create_social_media_performance():
    """Create social media post performance and audience interaction data"""
    np.random.seed(58)
    
    # Generate 100 posts across platforms
    platforms = ['Facebook', 'Instagram', 'Twitter', 'LinkedIn', 'TikTok']
    
    post_types = ['Image', 'Video', 'Text', 'Carousel', 'Story', 'Reel']
    
    data = []
    
    for i in range(100):
        platform = random.choice(platforms)
        post_type = random.choice(post_types)
        
        # Post time (hour of day)
        post_hour = np.random.randint(0, 24)
        
        # Engagement based on platform and post type
        if platform == 'Instagram' and post_type in ['Reel', 'Video']:
            base_impressions = np.random.randint(5000, 20000)
            engagement_rate = np.random.uniform(3, 6)
        elif platform == 'TikTok':
            base_impressions = np.random.randint(10000, 30000)
            engagement_rate = np.random.uniform(4, 8)
        elif platform == 'Facebook':
            base_impressions = np.random.randint(3000, 10000)
            engagement_rate = np.random.uniform(1, 3)
        elif platform == 'Twitter':
            base_impressions = np.random.randint(2000, 8000)
            engagement_rate = np.random.uniform(2, 4)
        else:  # LinkedIn
            base_impressions = np.random.randint(1000, 5000)
            engagement_rate = np.random.uniform(1.5, 3)
        
        # Time of day effect
        if 9 <= post_hour <= 17:  # Business hours
            impressions = int(base_impressions * np.random.uniform(1.2, 1.5))
        elif 18 <= post_hour <= 22:  # Evening peak
            impressions = int(base_impressions * np.random.uniform(1.3, 1.8))
        else:  # Off hours
            impressions = int(base_impressions * np.random.uniform(0.6, 0.9))
        
        # Calculate engagements
        engagement_rate = engagement_rate * np.random.uniform(0.8, 1.2)
        engagements = int(impressions * engagement_rate / 100)
        
        # Breakdown of engagements by type
        likes = int(engagements * np.random.uniform(0.5, 0.7))
        comments = int(engagements * np.random.uniform(0.1, 0.25))
        shares = int(engagements * np.random.uniform(0.05, 0.15))
        saves = int(engagements * np.random.uniform(0.02, 0.08))
        clicks = int(impressions * np.random.uniform(0.5, 2.0) / 100)
        
        # Video specific metrics
        video_view_rate = 0
        avg_watch_time = 0
        if post_type in ['Video', 'Reel']:
            video_view_rate = np.random.uniform(20, 60)
            avg_watch_time = np.random.uniform(15, 45)
        
        # Hashtags and mentions
        hashtag_count = np.random.randint(0, 10)
        mention_count = np.random.randint(0, 5)
        
        # Date for the post
        post_date = datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 90))
        
        # Content category
        categories = ['Educational', 'Entertainment', 'Promotional', 'Behind the Scenes', 'User Generated']
        category = random.choice(categories)
        
        # Virality score
        virality_score = (shares * 3 + comments * 2 + saves * 1.5) / impressions * 100 if impressions > 0 else 0
        
        data.append({
            'post_id': f'POST{1000 + i}',
            'platform': platform,
            'post_type': post_type,
            'post_date': post_date.strftime('%Y-%m-%d'),
            'post_hour': post_hour,
            'category': category,
            'impressions': impressions,
            'reach': int(impressions * np.random.uniform(0.7, 0.9)),
            'engagement_rate': round(engagement_rate, 2),
            'engagements': engagements,
            'likes': likes,
            'comments': comments,
            'shares': shares,
            'saves': saves,
            'clicks': clicks,
            'click_through_rate': round(clicks / impressions * 100, 2) if impressions > 0 else 0,
            'video_view_rate': round(video_view_rate, 2),
            'avg_watch_time_seconds': round(avg_watch_time, 1),
            'hashtag_count': hashtag_count,
            'mention_count': mention_count,
            'virality_score': round(virality_score, 3),
            'follower_growth': np.random.randint(-10, 50),
            'sentiment_score': np.random.uniform(0.6, 0.9)  # 0-1 scale
        })
    
    return pd.DataFrame(data)

def create_app_usage_data():
    """Create app usage data and user retention rates"""
    np.random.seed(59)
    
    # Generate 30 days of app usage data
    dates = []
    start_date = datetime(2024, 3, 1)
    for i in range(30):
        date = start_date + timedelta(days=i)
        dates.append(date.strftime('%Y-%m-%d'))
    
    # User cohorts based on signup date
    cohort_dates = ['2024-02-15', '2024-02-20', '2024-02-25', '2024-03-01', 
                    '2024-03-05', '2024-03-10', '2024-03-15']
    
    data = []
    
    # Daily metrics
    for i, date in enumerate(dates):
        # Daily active users
        dau = np.random.randint(5000, 15000)
        
        # Monthly active users
        mau = np.random.randint(30000, 50000)
        
        # Stickiness (DAU/MAU)
        stickiness = dau / mau * 100 if mau > 0 else 0
        
        # Session metrics
        sessions = np.random.randint(8000, 25000)
        avg_session_length = np.random.uniform(120, 600)  # seconds
        sessions_per_user = sessions / dau if dau > 0 else 0
        
        # Retention (simulated)
        day_since_signup = i % 30
        if day_since_signup == 0:
            retention_rate = 100
        else:
            retention_rate = 100 * np.exp(-0.1 * day_since_signup) * np.random.uniform(0.8, 1.2)
        
        # Feature usage
        feature_usage = {
            'messaging': np.random.uniform(40, 70),
            'profile_view': np.random.uniform(60, 90),
            'search': np.random.uniform(30, 60),
            'notifications': np.random.uniform(50, 80),
            'settings': np.random.uniform(10, 30)
        }
        
        # In-app events
        events_per_session = np.random.uniform(3, 8)
        conversion_rate = np.random.uniform(1, 5)
        
        # Crash and error rates
        crash_rate = np.random.uniform(0.1, 2.0)
        error_rate = np.random.uniform(0.5, 3.0)
        
        # Platform breakdown
        ios_users = int(dau * np.random.uniform(0.45, 0.55))
        android_users = dau - ios_users
        
        # User engagement score
        engagement_score = (stickiness * 0.3 + 
                           (avg_session_length / 600 * 100) * 0.2 +
                           sessions_per_user * 0.2 +
                           retention_rate * 0.3)
        
        data.append({
            'date': date,
            'day_since_start': i + 1,
            'daily_active_users': dau,
            'monthly_active_users': mau,
            'stickiness_percent': round(stickiness, 2),
            'total_sessions': sessions,
            'avg_session_length_seconds': round(avg_session_length, 1),
            'sessions_per_user': round(sessions_per_user, 2),
            'retention_rate_day': round(retention_rate, 2),
            'new_users': np.random.randint(200, 800),
            'churned_users': np.random.randint(100, 400),
            'messaging_usage_percent': round(feature_usage['messaging'], 2),
            'profile_view_usage_percent': round(feature_usage['profile_view'], 2),
            'search_usage_percent': round(feature_usage['search'], 2),
            'notification_usage_percent': round(feature_usage['notifications'], 2),
            'settings_usage_percent': round(feature_usage['settings'], 2),
            'events_per_session': round(events_per_session, 2),
            'conversion_rate': round(conversion_rate, 2),
            'crash_rate': round(crash_rate, 2),
            'error_rate': round(error_rate, 2),
            'ios_users': ios_users,
            'android_users': android_users,
            'engagement_score': round(engagement_score, 2),
            'app_store_rating': round(np.random.uniform(3.5, 4.8), 1),
            'app_launches': int(sessions * np.random.uniform(1.1, 1.3))
        })
    
    # Add cohort retention data
    cohort_data = []
    for cohort_date in cohort_dates:
        cohort_day = datetime.strptime(cohort_date, '%Y-%m-%d')
        for day_num in range(1, 31):
            retention = 100 * np.exp(-0.08 * day_num) * np.random.uniform(0.9, 1.1)
            cohort_data.append({
                'cohort_date': cohort_date,
                'day_number': day_num,
                'retention_percent': round(retention, 2),
                'cohort_size': np.random.randint(500, 2000)
            })
    
    return pd.DataFrame(data), pd.DataFrame(cohort_data)

def create_customer_support_data():
    """Create customer support ticket volumes and resolution times"""
    np.random.seed(60)
    
    # Generate 90 days of support data
    dates = []
    start_date = datetime(2024, 1, 1)
    for i in range(90):
        date = start_date + timedelta(days=i)
        dates.append(date.strftime('%Y-%m-%d'))
    
    days_of_week = [(start_date + timedelta(days=i)).strftime('%A') for i in range(90)]
    
    data = []
    
    for i, (date, day) in enumerate(zip(dates, days_of_week)):
        # Ticket volume patterns
        if day in ['Monday', 'Tuesday']:
            base_tickets = np.random.randint(150, 250)
        elif day in ['Wednesday', 'Thursday']:
            base_tickets = np.random.randint(120, 200)
        elif day == 'Friday':
            base_tickets = np.random.randint(100, 180)
        else:  # Weekend
            base_tickets = np.random.randint(50, 120)
        
        # Add some spikes
        if i % 14 == 0:  # Every 2 weeks
            base_tickets = int(base_tickets * np.random.uniform(1.3, 1.8))
        
        tickets_received = base_tickets
        tickets_resolved = int(base_tickets * np.random.uniform(0.9, 1.1))
        
        # Ticket categories
        categories = ['Technical', 'Billing', 'Account', 'Feature Request', 'General']
        category_distribution = np.random.dirichlet(np.ones(5)) * tickets_received
        
        # Channel distribution
        channels = ['Email', 'Chat', 'Phone', 'Social Media', 'Self-Service']
        channel_distribution = np.random.dirichlet(np.ones(5)) * tickets_received
        
        # Resolution times (hours)
        avg_resolution_time = np.random.uniform(2, 24)
        if day in ['Saturday', 'Sunday']:
            avg_resolution_time *= np.random.uniform(1.5, 2.0)  # Longer on weekends
        
        # First response time (hours)
        first_response_time = np.random.uniform(0.5, 4)
        
        # Customer satisfaction
        csat_score = np.random.uniform(3.5, 5.0)
        if avg_resolution_time > 12:
            csat_score -= np.random.uniform(0.5, 1.0)
        
        # Escalation rate
        escalation_rate = np.random.uniform(5, 15)
        
        # Agent metrics
        active_agents = np.random.randint(8, 15)
        tickets_per_agent = tickets_received / active_agents if active_agents > 0 else 0
        
        # Backlog
        if i == 0:
            backlog = np.random.randint(100, 200)
        else:
            previous_backlog = data[-1]['backlog']
            backlog = max(0, previous_backlog + tickets_received - tickets_resolved)
        
        data.append({
            'date': date,
            'day_of_week': day,
            'tickets_received': tickets_received,
            'tickets_resolved': tickets_resolved,
            'technical_tickets': int(category_distribution[0]),
            'billing_tickets': int(category_distribution[1]),
            'account_tickets': int(category_distribution[2]),
            'feature_request_tickets': int(category_distribution[3]),
            'general_tickets': int(category_distribution[4]),
            'email_tickets': int(channel_distribution[0]),
            'chat_tickets': int(channel_distribution[1]),
            'phone_tickets': int(channel_distribution[2]),
            'social_media_tickets': int(channel_distribution[3]),
            'self_service_tickets': int(channel_distribution[4]),
            'avg_resolution_time_hours': round(avg_resolution_time, 2),
            'first_response_time_hours': round(first_response_time, 2),
            'csat_score': round(csat_score, 1),
            'escalation_rate': round(escalation_rate, 2),
            'active_agents': active_agents,
            'tickets_per_agent': round(tickets_per_agent, 2),
            'backlog': backlog,
            'reopened_tickets': int(tickets_resolved * np.random.uniform(0.05, 0.15)),
            'sla_breaches': int(tickets_received * np.random.uniform(0.02, 0.08))
        })
    
    return pd.DataFrame(data)

def create_tech_survey_data():
    """Create survey responses about technology adoption and preferences"""
    np.random.seed(61)
    
    # Generate 200 survey responses
    respondent_ids = [f'R{2000 + i}' for i in range(200)]
    
    # Demographic data
    ages = np.random.randint(18, 70, 200)
    genders = np.random.choice(['Male', 'Female', 'Non-binary', 'Prefer not to say'], 
                              200, p=[0.48, 0.48, 0.02, 0.02])
    
    # Tech proficiency levels
    tech_proficiency = np.random.choice(['Beginner', 'Intermediate', 'Advanced', 'Expert'], 
                                        200, p=[0.2, 0.5, 0.25, 0.05])
    
    # Industry/sector
    industries = ['Technology', 'Healthcare', 'Finance', 'Education', 'Retail', 
                  'Manufacturing', 'Government', 'Other']
    
    data = []
    
    for i, rid in enumerate(respondent_ids):
        age = ages[i]
        gender = genders[i]
        proficiency = tech_proficiency[i]
        industry = random.choice(industries)
        
        # Device usage
        primary_device = random.choice(['Smartphone', 'Laptop', 'Desktop', 'Tablet', 'Multiple'])
        daily_screen_time = np.random.uniform(3, 12)
        
        # Technology adoption
        tech_adoption_score = np.random.randint(1, 11)  # 1-10 scale
        
        # AI/ML familiarity
        ai_familiarity = np.random.choice(['Not familiar', 'Somewhat familiar', 
                                           'Familiar', 'Very familiar'],
                                         p=[0.2, 0.4, 0.3, 0.1])
        
        # Cloud usage
        cloud_usage = random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Always'])
        
        # Security concerns
        security_concern = np.random.randint(1, 11)  # 1-10 scale
        
        # Social media usage
        social_media_hours = np.random.uniform(0.5, 4)
        
        # Preferred communication
        communication_preference = random.choice(['Email', 'Messaging App', 'Phone', 
                                                 'Video Call', 'In-person'])
        
        # E-commerce frequency
        ecommerce_frequency = random.choice(['Daily', 'Weekly', 'Monthly', 'Rarely', 'Never'])
        
        # Subscription services
        subscription_count = np.random.randint(0, 10)
        
        # Digital banking usage
        digital_banking = random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Always'])
        
        # Smart home devices
        smart_home_devices = np.random.randint(0, 8)
        
        # Work from home frequency
        wfh_frequency = np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Always'])
        
        # Privacy concerns
        privacy_concern = np.random.randint(1, 11)
        
        # Future tech interest
        future_tech_interest = np.random.randint(1, 11)
        
        # Calculate digital literacy score
        digital_literacy = 0
        if proficiency == 'Beginner':
            digital_literacy = np.random.randint(3, 6)
        elif proficiency == 'Intermediate':
            digital_literacy = np.random.randint(6, 8)
        elif proficiency == 'Advanced':
            digital_literacy = np.random.randint(8, 10)
        else:  # Expert
            digital_literacy = np.random.randint(9, 11)
        
        # Tech anxiety score (inverse of comfort)
        tech_anxiety = 10 - digital_literacy + np.random.randint(-2, 3)
        tech_anxiety = max(1, min(10, tech_anxiety))
        
        # Productivity impact
        tech_productivity = np.random.randint(6, 11)  # Mostly positive
        
        data.append({
            'respondent_id': rid,
            'age': age,
            'gender': gender,
            'tech_proficiency': proficiency,
            'industry': industry,
            'primary_device': primary_device,
            'daily_screen_time_hours': round(daily_screen_time, 1),
            'tech_adoption_score': tech_adoption_score,
            'ai_familiarity': ai_familiarity,
            'cloud_usage_frequency': cloud_usage,
            'security_concern_level': security_concern,
            'social_media_hours_daily': round(social_media_hours, 1),
            'communication_preference': communication_preference,
            'ecommerce_frequency': ecommerce_frequency,
            'subscription_count': subscription_count,
            'digital_banking_usage': digital_banking,
            'smart_home_devices': smart_home_devices,
            'work_from_home_frequency': wfh_frequency,
            'privacy_concern_level': privacy_concern,
            'future_tech_interest': future_tech_interest,
            'digital_literacy_score': digital_literacy,
            'tech_anxiety_score': tech_anxiety,
            'tech_productivity_impact': tech_productivity,
            'prefers_mobile_apps': random.choice(['Yes', 'No', 'Sometimes']),
            'uses_password_manager': random.choice(['Yes', 'No']),
            'regularly_updates_software': random.choice(['Always', 'Often', 'Sometimes', 'Rarely', 'Never']),
            'concerned_about_ai': random.choice(['Very concerned', 'Somewhat concerned', 
                                                'Neutral', 'Not very concerned', 'Not at all concerned'])
        })
    
    return pd.DataFrame(data)

def main():
    """Create all technology datasets and save to CSV files"""
    print("Creating technology and digital datasets...")
    
    # Create directory if it doesn't exist
    import os
    os.makedirs('data', exist_ok=True)
    
    # Create and save each dataset
    website_df = create_website_traffic_data()
    website_df.to_csv('data/website_traffic.csv', index=False)
    print(f"Created website_traffic.csv with {len(website_df)} rows")
    
    social_df = create_social_media_performance()
    social_df.to_csv('data/social_media_performance.csv', index=False)
    print(f"Created social_media_performance.csv with {len(social_df)} rows")
    
    app_usage_df, cohort_df = create_app_usage_data()
    app_usage_df.to_csv('data/app_usage.csv', index=False)
    cohort_df.to_csv('data/app_cohort_retention.csv', index=False)
    print(f"Created app_usage.csv with {len(app_usage_df)} rows")
    print(f"Created app_cohort_retention.csv with {len(cohort_df)} rows")
    
    support_df = create_customer_support_data()
    support_df.to_csv('data/customer_support.csv', index=False)
    print(f"Created customer_support.csv with {len(support_df)} rows")
    
    survey_df = create_tech_survey_data()
    survey_df.to_csv('data/tech_survey.csv', index=False)
    print(f"Created tech_survey.csv with {len(survey_df)} rows")
    
    print("\nAll datasets created successfully!")
    
