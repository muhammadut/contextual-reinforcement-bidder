"""
Exploratory Data Analysis for Training Data
Generates comprehensive Excel report analyzing extracted training data
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TrainingDataEDA:
    """
    Comprehensive EDA for training data with Excel output
    """

    def __init__(self, df: pd.DataFrame, config: dict, output_dir: Path):
        """
        Initialize EDA analyzer

        Args:
            df: Raw extracted training data
            config: Pipeline configuration
            output_dir: Directory to save EDA report
        """
        self.df = df.copy()
        self.config = config
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def generate_report(self):
        """
        Generate comprehensive EDA report as Excel file with multiple sheets
        """
        logger.info("="*60)
        logger.info("GENERATING TRAINING DATA EDA REPORT")
        logger.info("="*60)

        output_path = self.output_dir / f"training_eda_report_{self.timestamp}.xlsx"

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: Executive Summary
            self._write_executive_summary(writer)

            # Sheet 2: Affiliate Performance
            self._write_affiliate_analysis(writer)

            # Sheet 3: Affiliate Anomalies & Missing
            self._write_affiliate_anomalies(writer)

            # Sheet 4: Geographic Coverage
            self._write_geographic_analysis(writer)

            # Sheet 4: Bid Behavior Analysis
            self._write_bid_analysis(writer)

            # Sheet 5: Temporal Patterns
            self._write_temporal_analysis(writer)

            # Sheet 6: Data Quality Report
            self._write_data_quality(writer)

            # Sheet 7: Revenue & Profitability
            self._write_revenue_analysis(writer)

            # Sheet 8: Context Coverage (if available)
            if 'context_id' in self.df.columns:
                self._write_context_analysis(writer)

        logger.info(f"✅ EDA report saved to: {output_path}")
        logger.info(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")

        # Print key findings to console
        self._print_key_findings()

        return output_path

    def _write_executive_summary(self, writer):
        """Sheet 1: High-level overview of the dataset"""
        logger.info("📊 Generating executive summary...")

        # Basic stats
        total_rows = len(self.df)
        date_col = 'pinged_at' if 'pinged_at' in self.df.columns else self.df.select_dtypes(include=['datetime64']).columns[0]
        date_range = f"{self.df[date_col].min()} to {self.df[date_col].max()}"
        days_span = (self.df[date_col].max() - self.df[date_col].min()).days

        # Win stats
        total_wins = self.df['won'].sum()
        win_rate = self.df['won'].mean()

        # Revenue stats (only on won bids)
        won_df = self.df[self.df['won'] == 1]
        total_profit = won_df['expected_revenue'].sum() if 'expected_revenue' in won_df.columns else 0
        avg_profit_per_win = total_profit / len(won_df) if len(won_df) > 0 else 0

        # Affiliate stats
        n_affiliates_config = len(self.config['extraction'].get('affiliate_integration_ids', []))
        n_affiliates_data = self.df['affiliate_integration_id'].nunique()

        # Geographic stats - exclude NaN/None values from count
        # After geo merge: county_name, state_abbr, postal_code exist
        n_counties = self.df['county_name'].dropna().nunique() if 'county_name' in self.df.columns else 0
        n_postal_codes = self.df['postal_code'].dropna().nunique() if 'postal_code' in self.df.columns else 0
        n_states = self.df['state_abbr'].dropna().nunique() if 'state_abbr' in self.df.columns else 0

        summary_data = {
            'Metric': [
                'Dataset Overview',
                'Total Rows',
                'Date Range',
                'Days Span',
                'Category ID',
                '',
                'Performance Metrics',
                'Total Wins',
                'Overall Win Rate',
                'Wins per Day',
                '',
                'Revenue Metrics',
                'Total Profit (Won Bids)',
                'Average Profit per Win',
                'Average Profit per Day',
                '',
                'Affiliate Coverage',
                'Affiliates in Config',
                'Affiliates in Data',
                'Coverage %',
                'Missing Affiliates',
                '',
                'Geographic Coverage',
                'Unique States',
                'Unique Counties',
                'Unique Postal Codes',
                'Avg Postal Codes per County',
                '',
                'Data Quality',
                'Duplicate Rows',
                'Rows with Missing Values',
            ],
            'Value': [
                '',
                f"{total_rows:,}",
                date_range,
                f"{days_span} days",
                self.config['extraction'].get('category_id', 'N/A'),
                '',
                '',
                f"{int(total_wins):,}",
                f"{win_rate:.2%}",
                f"{total_wins/days_span:.1f}" if days_span > 0 else "N/A",
                '',
                '',
                f"${total_profit:,.2f}",
                f"${avg_profit_per_win:.2f}",
                f"${total_profit/days_span:.2f}" if days_span > 0 else "N/A",
                '',
                '',
                n_affiliates_config,
                n_affiliates_data,
                f"{(n_affiliates_data/n_affiliates_config*100):.1f}%" if n_affiliates_config > 0 else "N/A",
                n_affiliates_config - n_affiliates_data if n_affiliates_config > 0 else "N/A",
                '',
                '',
                n_states,
                f"{n_counties:,}",
                f"{n_postal_codes:,}",
                f"{n_postal_codes/n_counties:.1f}" if n_counties > 0 else "N/A",
                '',
                '',
                self.df.duplicated().sum(),
                self.df.isnull().any(axis=1).sum(),
            ]
        }

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)

    def _write_affiliate_analysis(self, writer):
        """Sheet 2: Per-affiliate performance metrics"""
        logger.info("📊 Analyzing affiliate performance...")

        affiliate_stats = []

        for affiliate_id in sorted(self.df['affiliate_integration_id'].unique()):
            aff_df = self.df[self.df['affiliate_integration_id'] == affiliate_id]
            aff_won = aff_df[aff_df['won'] == 1]

            stats = {
                'Affiliate ID': int(affiliate_id),
                'Total Volume': len(aff_df),
                'Total Wins': int(aff_df['won'].sum()),
                'Win Rate': aff_df['won'].mean(),
                'Total Profit': aff_won['expected_revenue'].sum() if 'expected_revenue' in aff_won.columns else 0,
                'Avg Profit per Win': aff_won['expected_revenue'].mean() if 'expected_revenue' in aff_won.columns and len(aff_won) > 0 else 0,
                'Avg Bid Value': aff_df['final_modified'].mean() if 'final_modified' in aff_df.columns else 0,
                'Unique Counties': aff_df['county_name'].dropna().nunique() if 'county_name' in aff_df.columns else 0,
                'Unique Postal Codes': aff_df['postal_code'].dropna().nunique() if 'postal_code' in aff_df.columns else 0,
                'Wins per Postal Code': (aff_df['won'].sum() / aff_df['postal_code'].dropna().nunique()) if 'postal_code' in aff_df.columns and aff_df['postal_code'].dropna().nunique() > 0 else 0,
                'Avg Wins per Day': aff_df['won'].sum() / ((aff_df[aff_df.select_dtypes(include=['datetime64']).columns[0]].max() - aff_df[aff_df.select_dtypes(include=['datetime64']).columns[0]].min()).days + 1) if len(aff_df.select_dtypes(include=['datetime64']).columns) > 0 else 0,
            }

            # Add bid modification stats if available
            if 'bid_modification_variant_id' in aff_df.columns:
                config_rate = (aff_df['bid_modification_variant_id'] == 'configuration').mean()
                stats['Configuration Rate'] = config_rate

            affiliate_stats.append(stats)

        affiliate_df = pd.DataFrame(affiliate_stats)

        # Sort by total wins descending
        affiliate_df = affiliate_df.sort_values('Total Wins', ascending=False)

        # Add summary row
        summary_row = {
            'Affiliate ID': 'TOTAL/AVG',
            'Total Volume': affiliate_df['Total Volume'].sum(),
            'Total Wins': affiliate_df['Total Wins'].sum(),
            'Win Rate': affiliate_df['Win Rate'].mean(),
            'Total Profit': affiliate_df['Total Profit'].sum(),
            'Avg Profit per Win': affiliate_df['Avg Profit per Win'].mean(),
            'Avg Bid Value': affiliate_df['Avg Bid Value'].mean(),
            'Unique Counties': affiliate_df['Unique Counties'].sum(),
            'Unique Postal Codes': affiliate_df['Unique Postal Codes'].sum(),
            'Wins per Postal Code': affiliate_df['Wins per Postal Code'].mean(),
            'Avg Wins per Day': affiliate_df['Avg Wins per Day'].mean(),
        }

        if 'Configuration Rate' in affiliate_df.columns:
            summary_row['Configuration Rate'] = affiliate_df['Configuration Rate'].mean()

        affiliate_df = pd.concat([affiliate_df, pd.DataFrame([summary_row])], ignore_index=True)

        affiliate_df.to_excel(writer, sheet_name='Affiliate Performance', index=False)

    def _write_affiliate_anomalies(self, writer):
        """Sheet 3: Affiliate anomalies and missing partners"""
        logger.info("📊 Analyzing affiliate anomalies...")

        sections = []

        # Section 1: Missing Affiliates
        config_affiliate_ids = set(self.config['extraction'].get('affiliate_integration_ids', []))
        data_affiliate_ids = set(self.df['affiliate_integration_id'].unique())
        missing_affiliates = config_affiliate_ids - data_affiliate_ids

        sections.append({'Section': 'MISSING AFFILIATES', 'Detail': '', 'Value': ''})
        sections.append({'Section': 'Total in Config', 'Detail': '', 'Value': len(config_affiliate_ids)})
        sections.append({'Section': 'Total in Data', 'Detail': '', 'Value': len(data_affiliate_ids)})
        sections.append({'Section': 'Missing Count', 'Detail': '', 'Value': len(missing_affiliates)})
        sections.append({'Section': '', 'Detail': '', 'Value': ''})

        if missing_affiliates:
            sections.append({'Section': 'Missing Affiliate IDs:', 'Detail': '', 'Value': ''})
            for aff_id in sorted(missing_affiliates):
                sections.append({'Section': '', 'Detail': f'Affiliate {aff_id}', 'Value': 'NOT FOUND IN DATA'})
        else:
            sections.append({'Section': '', 'Detail': 'All configured affiliates present in data', 'Value': '✓'})

        sections.append({'Section': '', 'Detail': '', 'Value': ''})

        # Section 2: Anomalous Win Rates
        sections.append({'Section': 'ANOMALOUS WIN RATES', 'Detail': '', 'Value': ''})
        sections.append({'Section': '', 'Detail': 'Partners with unusual win rates', 'Value': ''})
        sections.append({'Section': '', 'Detail': '', 'Value': ''})

        anomalies = []
        for affiliate_id in data_affiliate_ids:
            aff_df = self.df[self.df['affiliate_integration_id'] == affiliate_id]
            win_rate = aff_df['won'].mean()
            total_wins = aff_df['won'].sum()

            # Flag anomalies
            reasons = []
            if win_rate >= 1.0:
                reasons.append("100% win rate")
            elif win_rate >= 0.80:
                reasons.append(f"Very high win rate ({win_rate:.1%})")
            elif win_rate <= 0.001 and len(aff_df) > 100:
                reasons.append(f"Very low win rate ({win_rate:.3%})")

            if total_wins == 0 and len(aff_df) > 50:
                reasons.append("Zero wins")
            elif total_wins < 5 and len(aff_df) > 100:
                reasons.append(f"Very few wins ({int(total_wins)})")

            if reasons:
                anomalies.append({
                    'Affiliate ID': int(affiliate_id),
                    'Total Volume': len(aff_df),
                    'Total Wins': int(total_wins),
                    'Win Rate': win_rate,
                    'Flags': ' | '.join(reasons)
                })

        if anomalies:
            # Sort by win rate descending
            anomalies_df = pd.DataFrame(anomalies).sort_values('Win Rate', ascending=False)

            # Add to sections
            for _, row in anomalies_df.iterrows():
                sections.append({
                    'Section': f"Affiliate {row['Affiliate ID']}",
                    'Detail': f"Volume: {row['Total Volume']:,}, Wins: {row['Total Wins']:,}, WR: {row['Win Rate']:.2%}",
                    'Value': row['Flags']
                })
        else:
            sections.append({'Section': '', 'Detail': 'No significant anomalies detected', 'Value': '✓'})

        sections.append({'Section': '', 'Detail': '', 'Value': ''})

        # Section 3: Low Sample Affiliates
        sections.append({'Section': 'LOW SAMPLE AFFILIATES', 'Detail': '', 'Value': ''})
        sections.append({'Section': '', 'Detail': 'Affiliates with < 100 samples', 'Value': ''})
        sections.append({'Section': '', 'Detail': '', 'Value': ''})

        low_sample_affiliates = []
        for affiliate_id in data_affiliate_ids:
            aff_df = self.df[self.df['affiliate_integration_id'] == affiliate_id]
            if len(aff_df) < 100:
                low_sample_affiliates.append({
                    'Affiliate ID': int(affiliate_id),
                    'Total Volume': len(aff_df),
                    'Total Wins': int(aff_df['won'].sum()),
                    'Win Rate': aff_df['won'].mean()
                })

        if low_sample_affiliates:
            low_sample_df = pd.DataFrame(low_sample_affiliates).sort_values('Total Volume', ascending=True)
            for _, row in low_sample_df.iterrows():
                sections.append({
                    'Section': f"Affiliate {row['Affiliate ID']}",
                    'Detail': f"Volume: {row['Total Volume']:,}, Wins: {row['Total Wins']:,}",
                    'Value': f"WR: {row['Win Rate']:.2%}"
                })
        else:
            sections.append({'Section': '', 'Detail': 'All affiliates have sufficient samples', 'Value': '✓'})

        anomalies_df = pd.DataFrame(sections)
        anomalies_df.to_excel(writer, sheet_name='Affiliate Anomalies', index=False)

    def _write_geographic_analysis(self, writer):
        """Sheet 4: Geographic coverage analysis"""
        logger.info("📊 Analyzing geographic coverage...")

        if 'county_name' not in self.df.columns:
            logger.warning("County column not found, skipping geographic analysis")
            return

        geo_stats = []

        # Group by county
        for county in self.df['county_name'].dropna().unique():
            county_df = self.df[self.df['county_name'] == county]
            county_won = county_df[county_df['won'] == 1]

            stats = {
                'County': county,
                'State': county_df['state_abbr'].mode()[0] if 'state_abbr' in county_df.columns and len(county_df['state_abbr'].mode()) > 0 else 'N/A',
                'Total Volume': len(county_df),
                'Total Wins': int(county_df['won'].sum()),
                'Win Rate': county_df['won'].mean(),
                'Unique Postal Codes': county_df['postal_code'].dropna().nunique() if 'postal_code' in county_df.columns else 0,
                'N Affiliates': county_df['affiliate_integration_id'].nunique(),
                'Avg Wins per Postal Code': county_df['won'].sum() / county_df['postal_code'].dropna().nunique() if 'postal_code' in county_df.columns and county_df['postal_code'].dropna().nunique() > 0 else 0,
                'Total Profit': county_won['expected_revenue'].sum() if 'expected_revenue' in county_won.columns else 0,
            }

            geo_stats.append(stats)

        geo_df = pd.DataFrame(geo_stats)
        geo_df = geo_df.sort_values('Total Wins', ascending=False)

        # Add summary
        summary = {
            'County': 'TOTAL/AVG',
            'State': f"{geo_df['State'].nunique()} states",
            'Total Volume': geo_df['Total Volume'].sum(),
            'Total Wins': geo_df['Total Wins'].sum(),
            'Win Rate': geo_df['Win Rate'].mean(),
            'Unique Postal Codes': geo_df['Unique Postal Codes'].sum(),
            'N Affiliates': geo_df['N Affiliates'].mean(),
            'Avg Wins per Postal Code': geo_df['Avg Wins per Postal Code'].mean(),
            'Total Profit': geo_df['Total Profit'].sum(),
        }

        geo_df = pd.concat([geo_df, pd.DataFrame([summary])], ignore_index=True)

        geo_df.to_excel(writer, sheet_name='Geographic Coverage', index=False)

    def _write_bid_analysis(self, writer):
        """Sheet 4: Bid behavior and modification analysis"""
        logger.info("📊 Analyzing bid behavior...")

        bid_stats_data = {
            'Metric': [],
            'Value': []
        }

        # Overall bid statistics
        if 'final_modified' in self.df.columns:
            bid_stats_data['Metric'].extend([
                'Bid Statistics',
                'Mean Bid Value',
                'Median Bid Value',
                'Std Dev Bid Value',
                'Min Bid Value',
                'Max Bid Value',
                '25th Percentile',
                '75th Percentile',
                '',
                'Winning Bid Statistics',
                'Mean Winning Bid',
                'Median Winning Bid',
                'Std Dev Winning Bid',
            ])

            won_df = self.df[self.df['won'] == 1]

            bid_stats_data['Value'].extend([
                '',
                f"{self.df['final_modified'].mean():.2f}%",
                f"{self.df['final_modified'].median():.2f}%",
                f"{self.df['final_modified'].std():.2f}%",
                f"{self.df['final_modified'].min():.2f}%",
                f"{self.df['final_modified'].max():.2f}%",
                f"{self.df['final_modified'].quantile(0.25):.2f}%",
                f"{self.df['final_modified'].quantile(0.75):.2f}%",
                '',
                '',
                f"{won_df['final_modified'].mean():.2f}%" if len(won_df) > 0 else "N/A",
                f"{won_df['final_modified'].median():.2f}%" if len(won_df) > 0 else "N/A",
                f"{won_df['final_modified'].std():.2f}%" if len(won_df) > 0 else "N/A",
            ])

        # Bid modification variant analysis
        if 'bid_modification_variant_id' in self.df.columns:
            bid_stats_data['Metric'].append('')
            bid_stats_data['Value'].append('')
            bid_stats_data['Metric'].append('Bid Modification Variants')
            bid_stats_data['Value'].append('')

            for variant in self.df['bid_modification_variant_id'].unique():
                variant_df = self.df[self.df['bid_modification_variant_id'] == variant]
                count = len(variant_df)
                pct = count / len(self.df) * 100
                win_rate = variant_df['won'].mean()

                bid_stats_data['Metric'].append(f"  {variant}")
                bid_stats_data['Value'].append(f"{count:,} ({pct:.1f}%) - WR: {win_rate:.2%}")

        bid_df = pd.DataFrame(bid_stats_data)
        bid_df.to_excel(writer, sheet_name='Bid Behavior', index=False)

    def _write_temporal_analysis(self, writer):
        """Sheet 5: Temporal patterns and trends"""
        logger.info("📊 Analyzing temporal patterns...")

        date_col = 'pinged_at' if 'pinged_at' in self.df.columns else self.df.select_dtypes(include=['datetime64']).columns[0]

        # Ensure datetime
        self.df[date_col] = pd.to_datetime(self.df[date_col])

        # Daily aggregation
        self.df['date'] = self.df[date_col].dt.date
        daily_stats = self.df.groupby('date').agg({
            'won': ['count', 'sum', 'mean'],
            'expected_revenue': lambda x: x[self.df.loc[x.index, 'won'] == 1].sum() if 'expected_revenue' in self.df.columns else 0
        }).reset_index()

        daily_stats.columns = ['Date', 'Total Volume', 'Total Wins', 'Win Rate', 'Total Profit']
        daily_stats = daily_stats.sort_values('Date')

        daily_stats.to_excel(writer, sheet_name='Temporal Patterns', index=False)

    def _write_data_quality(self, writer):
        """Sheet 6: Data quality assessment"""
        logger.info("📊 Assessing data quality...")

        quality_data = []

        # Missing values per column
        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            missing_pct = missing_count / len(self.df) * 100

            quality_data.append({
                'Column': col,
                'Missing Count': missing_count,
                'Missing %': missing_pct,
                'Dtype': str(self.df[col].dtype),
                'Unique Values': self.df[col].nunique(),
            })

        quality_df = pd.DataFrame(quality_data)
        quality_df = quality_df.sort_values('Missing Count', ascending=False)

        quality_df.to_excel(writer, sheet_name='Data Quality', index=False)

    def _write_revenue_analysis(self, writer):
        """Sheet 7: Revenue and profitability deep dive"""
        logger.info("📊 Analyzing revenue and profitability...")

        if 'expected_revenue' not in self.df.columns:
            logger.warning("Revenue column not found, skipping revenue analysis")
            return

        won_df = self.df[self.df['won'] == 1].copy()

        revenue_stats = {
            'Metric': [
                'Overall Revenue Stats',
                'Total Profit (Won Bids)',
                'Mean Profit per Win',
                'Median Profit per Win',
                'Std Dev Profit',
                'Min Profit',
                'Max Profit',
                '',
                'Revenue Distribution',
                'Top 10% Wins Revenue',
                'Top 25% Wins Revenue',
                'Bottom 50% Wins Revenue',
                '',
                'Profitability Analysis',
                'Wins with Profit > $50',
                'Wins with Profit > $100',
                'Wins with Profit < $20',
            ],
            'Value': [
                '',
                f"${won_df['expected_revenue'].sum():,.2f}",
                f"${won_df['expected_revenue'].mean():.2f}",
                f"${won_df['expected_revenue'].median():.2f}",
                f"${won_df['expected_revenue'].std():.2f}",
                f"${won_df['expected_revenue'].min():.2f}",
                f"${won_df['expected_revenue'].max():.2f}",
                '',
                '',
                f"${won_df.nlargest(int(len(won_df)*0.1), 'expected_revenue')['expected_revenue'].sum():,.2f}",
                f"${won_df.nlargest(int(len(won_df)*0.25), 'expected_revenue')['expected_revenue'].sum():,.2f}",
                f"${won_df.nsmallest(int(len(won_df)*0.5), 'expected_revenue')['expected_revenue'].sum():,.2f}",
                '',
                '',
                f"{(won_df['expected_revenue'] > 50).sum():,} ({(won_df['expected_revenue'] > 50).mean():.1%})",
                f"{(won_df['expected_revenue'] > 100).sum():,} ({(won_df['expected_revenue'] > 100).mean():.1%})",
                f"{(won_df['expected_revenue'] < 20).sum():,} ({(won_df['expected_revenue'] < 20).mean():.1%})",
            ]
        }

        revenue_df = pd.DataFrame(revenue_stats)
        revenue_df.to_excel(writer, sheet_name='Revenue Analysis', index=False)

    def _write_context_analysis(self, writer):
        """Sheet 8: Context coverage analysis (if context_id exists)"""
        logger.info("📊 Analyzing context coverage...")

        context_stats = self.df.groupby('context_id').agg({
            'won': ['count', 'sum', 'mean'],
            'affiliate_integration_id': 'first',
            'expected_revenue': lambda x: x[self.df.loc[x.index, 'won'] == 1].sum() if 'expected_revenue' in self.df.columns else 0
        }).reset_index()

        context_stats.columns = ['Context ID', 'Total Volume', 'Total Wins', 'Win Rate', 'Affiliate ID', 'Total Profit']
        context_stats = context_stats.sort_values('Total Wins', ascending=False)

        # Add summary
        summary = {
            'Context ID': 'TOTAL/AVG',
            'Total Volume': context_stats['Total Volume'].sum(),
            'Total Wins': context_stats['Total Wins'].sum(),
            'Win Rate': context_stats['Win Rate'].mean(),
            'Affiliate ID': f"{context_stats['Affiliate ID'].nunique()} affiliates",
            'Total Profit': context_stats['Total Profit'].sum(),
        }

        context_stats = pd.concat([context_stats, pd.DataFrame([summary])], ignore_index=True)

        context_stats.to_excel(writer, sheet_name='Context Coverage', index=False)

    def _print_key_findings(self):
        """Print key findings to console for quick review"""
        logger.info("\n" + "="*60)
        logger.info("KEY FINDINGS SUMMARY")
        logger.info("="*60)

        # Dataset size
        logger.info(f"📊 Dataset: {len(self.df):,} rows")

        # Win rate
        win_rate = self.df['won'].mean()
        logger.info(f"🎯 Overall Win Rate: {win_rate:.2%}")

        # Affiliates
        n_affiliates = self.df['affiliate_integration_id'].nunique()
        logger.info(f"👥 Affiliates in Data: {n_affiliates}")

        # Top performer
        top_affiliate = self.df.groupby('affiliate_integration_id')['won'].sum().idxmax()
        top_wins = self.df.groupby('affiliate_integration_id')['won'].sum().max()
        logger.info(f"🏆 Top Performer: Affiliate {int(top_affiliate)} ({int(top_wins):,} wins)")

        # Revenue
        if 'expected_revenue' in self.df.columns:
            total_profit = self.df[self.df['won'] == 1]['expected_revenue'].sum()
            logger.info(f"💰 Total Profit: ${total_profit:,.2f}")

        # Geographic coverage
        if 'county_name' in self.df.columns:
            n_counties = self.df['county_name'].dropna().nunique()
            logger.info(f"🗺️  Geographic Coverage: {n_counties:,} counties")

        logger.info("="*60 + "\n")


def run_eda_if_enabled(df: pd.DataFrame, config: dict, output_dir: Path) -> bool:
    """
    Run EDA if enabled in config

    Args:
        df: Raw extracted training data
        config: Pipeline configuration
        output_dir: Directory to save EDA report

    Returns:
        True if EDA was run, False otherwise
    """
    if not config.get('eda', {}).get('enabled', False):
        logger.info("EDA disabled in config, skipping...")
        return False

    try:
        eda = TrainingDataEDA(df, config, output_dir)
        eda.generate_report()
        return True
    except Exception as e:
        logger.error(f"EDA generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
