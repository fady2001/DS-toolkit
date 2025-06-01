from datetime import datetime
from pathlib import Path
from typing import Optional
import warnings

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer

from config import FIGURES_DIR, RAW_DATA_DIR

warnings.filterwarnings("ignore")

app = typer.Typer()

# Set visualization style
plt.style.use("default")
sns.set_palette("deep")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10


class EDAPlotter:
    """Comprehensive EDA plotting class for store sales data"""

    def __init__(self, figures_dir: Path):
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # Day names for better interpretation
        self.day_names = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]

    def load_data(self, data_dir: Path):
        """Load all required datasets"""
        logger.info("Loading datasets...")

        self.train_df = pd.read_csv(data_dir / "train.csv", parse_dates=["date"])
        self.holiday_df = pd.read_csv(data_dir / "holidays_events.csv", parse_dates=["date"])
        self.oil_df = pd.read_csv(data_dir / "oil.csv", parse_dates=["date"])
        self.stores_df = pd.read_csv(data_dir / "stores.csv")
        self.transactions_df = pd.read_csv(data_dir / "transactions.csv", parse_dates=["date"])

        logger.info("Data loaded successfully")

    def prepare_data(self):
        """Prepare data for analysis"""
        logger.info("Preparing data for analysis...")

        # Create temporal features
        self.train_df["year"] = self.train_df["date"].dt.year
        self.train_df["month"] = self.train_df["date"].dt.month
        self.train_df["day"] = self.train_df["date"].dt.day
        self.train_df["dayofweek"] = self.train_df["date"].dt.dayofweek
        self.train_df["day_name"] = self.train_df["dayofweek"].map(lambda x: self.day_names[x])
        self.train_df["log_sales"] = np.log1p(self.train_df["sales"])

        # Store performance analysis
        store_performance = (
            self.train_df.groupby("store_nbr")
            .agg({"sales": ["sum", "mean", "std", "count"], "onpromotion": "mean"})
            .round(2)
        )

        store_performance.columns = [
            "Total_Sales",
            "Avg_Sales",
            "Sales_Std",
            "Records_Count",
            "Promo_Rate",
        ]
        store_performance = store_performance.reset_index()

        # Merge with store information
        self.store_analysis = store_performance.merge(
            self.stores_df, left_on="store_nbr", right_on="store_nbr", how="left"
        )

        # Product family performance
        family_performance = (
            self.train_df.groupby("family")
            .agg({"sales": ["sum", "mean", "std", "count"], "onpromotion": "mean"})
            .round(2)
        )

        family_performance.columns = [
            "Total_Sales",
            "Avg_Sales",
            "Sales_Std",
            "Records_Count",
            "Promo_Rate",
        ]
        self.family_performance = family_performance.reset_index().sort_values(
            "Total_Sales", ascending=False
        )

        # Create holiday indicator
        self.train_df["is_holiday"] = (
            self.train_df["date"].isin(self.holiday_df["date"]).astype(int)
        )

        # Interpolate oil prices
        self.oil_df["interpolated_dcoilwtico"] = self.oil_df["dcoilwtico"].interpolate(
            method="polynomial", order=2
        )

        logger.info("Data preparation complete")

    def plot_sales_distribution(self):
        """Plot sales distribution"""
        logger.info("Creating sales distribution plots...")

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Sales distribution
        sns.histplot(self.train_df["sales"], bins=50, kde=True, ax=axes[0], color="blue")
        axes[0].set_title("Sales Distribution", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("Sales")
        axes[0].set_ylabel("Frequency")

        # Log sales distribution
        sns.histplot(self.train_df["log_sales"], bins=50, kde=True, ax=axes[1], color="green")
        axes[1].set_title("Log Sales Distribution", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Log Sales")
        axes[1].set_ylabel("Frequency")

        plt.tight_layout()
        plt.savefig(self.figures_dir / "sales_distribution.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_temporal_patterns(self):
        """Plot temporal sales patterns"""
        logger.info("Creating temporal pattern plots...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Daily sales trend
        daily_sales = self.train_df.groupby("date")["sales"].sum()
        axes[0, 0].plot(daily_sales.index, daily_sales.values)
        axes[0, 0].set_title("Daily Sales Trend", fontsize=14, fontweight="bold")
        axes[0, 0].set_xlabel("Date")
        axes[0, 0].set_ylabel("Total Sales")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # Monthly sales pattern
        monthly_sales = self.train_df.groupby("month")["sales"].mean()
        axes[0, 1].bar(monthly_sales.index, monthly_sales.values, color="skyblue")
        axes[0, 1].set_title("Average Sales by Month", fontsize=14, fontweight="bold")
        axes[0, 1].set_xlabel("Month")
        axes[0, 1].set_ylabel("Average Sales")
        axes[0, 1].set_xticks(range(1, 13))

        # Day of week pattern
        dow_sales = self.train_df.groupby("day_name")["sales"].mean().reindex(self.day_names)
        axes[1, 0].bar(range(len(dow_sales)), dow_sales.values, color="lightcoral")
        axes[1, 0].set_title("Average Sales by Day of Week", fontsize=14, fontweight="bold")
        axes[1, 0].set_xlabel("Day of Week")
        axes[1, 0].set_ylabel("Average Sales")
        axes[1, 0].set_xticks(range(len(self.day_names)))
        axes[1, 0].set_xticklabels(self.day_names, rotation=45)

        # Yearly sales trend
        yearly_sales = self.train_df.groupby("year")["sales"].sum()
        axes[1, 1].bar(yearly_sales.index, yearly_sales.values, color="lightgreen")
        axes[1, 1].set_title("Total Sales by Year", fontsize=14, fontweight="bold")
        axes[1, 1].set_xlabel("Year")
        axes[1, 1].set_ylabel("Total Sales")

        plt.tight_layout()
        plt.savefig(self.figures_dir / "temporal_patterns.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_sales_by_factors(self):
        """Plot sales by various factors"""
        logger.info("Creating sales by factors plots...")

        fig, axes = plt.subplots(3, 2, figsize=(16, 18))

        # Sales by store number
        store_sales = (
            self.train_df.groupby("store_nbr")["sales"].mean().sort_values(ascending=False)
        )
        axes[0, 0].bar(store_sales.index, store_sales.values, color="purple")
        axes[0, 0].set_title("Average Sales by Store Number", fontsize=14, fontweight="bold")
        axes[0, 0].set_xlabel("Store Number")
        axes[0, 0].set_ylabel("Average Sales")

        # Store cluster
        cluster_sales = (
            self.store_analysis.groupby("cluster")["Avg_Sales"].mean().sort_values(ascending=False)
        )
        axes[0, 1].bar(cluster_sales.index.astype(str), cluster_sales.values, color="lightgreen")
        axes[0, 1].set_title("Average Sales by Store Cluster", fontsize=14, fontweight="bold")
        axes[0, 1].set_xlabel("Store Cluster")
        axes[0, 1].set_ylabel("Average Sales")

        # Store type
        type_sales = (
            self.store_analysis.groupby("type")["Avg_Sales"].mean().sort_values(ascending=False)
        )
        axes[1, 0].bar(type_sales.index, type_sales.values, color="lightblue")
        axes[1, 0].set_title("Average Sales by Store Type", fontsize=14, fontweight="bold")
        axes[1, 0].set_xlabel("Store Type")
        axes[1, 0].set_ylabel("Average Sales")

        # Top cities
        city_sales = (
            self.store_analysis.groupby("city")["Total_Sales"]
            .sum()
            .sort_values(ascending=False)
            .head(15)
        )
        axes[1, 1].barh(range(len(city_sales)), city_sales.values)
        axes[1, 1].set_yticks(range(len(city_sales)))
        axes[1, 1].set_yticklabels(city_sales.index)
        axes[1, 1].set_title("Top 15 Cities by Total Sales", fontsize=14, fontweight="bold")
        axes[1, 1].set_xlabel("Total Sales")

        # Promotion effect
        promo_sales_binary = self.train_df.groupby(self.train_df["onpromotion"] > 0)[
            "sales"
        ].mean()
        axes[2, 0].bar(["No Promotion", "Promotion"], promo_sales_binary.values, color="coral")
        axes[2, 0].set_title(
            "Average Sales with/without Promotion", fontsize=14, fontweight="bold"
        )
        axes[2, 0].set_ylabel("Average Sales")
        axes[2, 0].set_xlabel("Promotion Status")

        # Sales by family
        family_sales = self.train_df.groupby("family")["sales"].mean().sort_values(ascending=True)
        axes[2, 1].barh(family_sales.index, family_sales.values, color="orange")
        axes[2, 1].set_title("Average Sales by Family", fontsize=14, fontweight="bold")
        axes[2, 1].set_xlabel("Average Sales")
        axes[2, 1].set_ylabel("Family")

        plt.tight_layout()
        plt.savefig(self.figures_dir / "sales_by_factors.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_store_analysis(self):
        """Plot store performance analysis"""
        logger.info("Creating store analysis plots...")

        fig, axes = plt.subplots(2, 1, figsize=(16, 12))

        # Top 10 stores
        top_10_stores = self.store_analysis.nlargest(10, "Total_Sales")
        axes[0].barh(
            top_10_stores["store_nbr"].astype(str), top_10_stores["Total_Sales"], color="purple"
        )
        axes[0].set_title("Top 10 Stores by Total Sales", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("Total Sales")
        axes[0].set_ylabel("Store Number")

        # Bottom 10 stores
        bottom_10_stores = self.store_analysis.nsmallest(10, "Total_Sales")
        axes[1].barh(
            bottom_10_stores["store_nbr"].astype(str),
            bottom_10_stores["Total_Sales"],
            color="orange",
        )
        axes[1].set_title("Bottom 10 Stores by Total Sales", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Total Sales")
        axes[1].set_ylabel("Store Number")

        plt.tight_layout()
        plt.savefig(self.figures_dir / "store_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_family_analysis(self):
        """Plot product family analysis"""
        logger.info("Creating product family analysis plots...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Top 15 families by total sales
        top_families = self.family_performance.head(15)
        axes[0, 0].barh(range(len(top_families)), top_families["Total_Sales"])
        axes[0, 0].set_yticks(range(len(top_families)))
        axes[0, 0].set_yticklabels(top_families["family"])
        axes[0, 0].set_title(
            "Top 15 Product Families by Total Sales", fontsize=14, fontweight="bold"
        )
        axes[0, 0].set_xlabel("Total Sales")

        # Average sales per family
        top_avg_families = self.family_performance.nlargest(15, "Avg_Sales")
        axes[0, 1].barh(range(len(top_avg_families)), top_avg_families["Avg_Sales"])
        axes[0, 1].set_yticks(range(len(top_avg_families)))
        axes[0, 1].set_yticklabels(top_avg_families["family"])
        axes[0, 1].set_title(
            "Top 15 Product Families by Average Sales", fontsize=14, fontweight="bold"
        )
        axes[0, 1].set_xlabel("Average Sales")

        # Promotion rate by family
        top_promo_families = self.family_performance.nlargest(15, "Promo_Rate")
        axes[1, 0].barh(range(len(top_promo_families)), top_promo_families["Promo_Rate"])
        axes[1, 0].set_yticks(range(len(top_promo_families)))
        axes[1, 0].set_yticklabels(top_promo_families["family"])
        axes[1, 0].set_title(
            "Top 15 Product Families by Promotion Rate", fontsize=14, fontweight="bold"
        )
        axes[1, 0].set_xlabel("Promotion Rate")

        # Sales volatility
        volatile_families = self.family_performance.nlargest(15, "Sales_Std")
        axes[1, 1].barh(range(len(volatile_families)), volatile_families["Sales_Std"])
        axes[1, 1].set_yticks(range(len(volatile_families)))
        axes[1, 1].set_yticklabels(volatile_families["family"])
        axes[1, 1].set_title(
            "Top 15 Most Volatile Product Families", fontsize=14, fontweight="bold"
        )
        axes[1, 1].set_xlabel("Sales Standard Deviation")

        plt.tight_layout()
        plt.savefig(self.figures_dir / "family_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_oil_analysis(self):
        """Plot oil price analysis"""
        logger.info("Creating oil price analysis plots...")

        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        # Oil price trends
        axes[0].plot(
            self.oil_df["date"],
            self.oil_df["dcoilwtico"],
            label="Original Oil Price",
            alpha=0.7,
            color="blue",
        )
        axes[0].plot(
            self.oil_df["date"],
            self.oil_df["interpolated_dcoilwtico"],
            label="Interpolated Oil Price",
            color="orange",
        )
        axes[0].set_title(
            "Oil Price Over Time (Original vs Interpolated)", fontsize=14, fontweight="bold"
        )
        axes[0].set_xlabel("Date")
        axes[0].set_ylabel("Oil Price (USD)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Sales vs Oil price correlation
        daily_sales_oil = self.train_df.groupby("date")["sales"].sum().reset_index()
        sales_oil_merged = daily_sales_oil.merge(
            self.oil_df[["date", "interpolated_dcoilwtico"]], on="date", how="left"
        )

        axes[1].scatter(
            sales_oil_merged["interpolated_dcoilwtico"], sales_oil_merged["sales"], alpha=0.6
        )
        axes[1].set_title("Daily Sales vs Oil Price", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Oil Price (USD)")
        axes[1].set_ylabel("Daily Total Sales")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.figures_dir / "oil_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_holiday_analysis(self):
        """Plot holiday analysis"""
        logger.info("Creating holiday analysis plots...")

        # Holiday impact
        holiday_impact = (
            self.train_df.groupby("is_holiday")["sales"]
            .agg(["mean", "median", "sum", "count"])
            .round(2)
        )
        holiday_impact.index = ["Non-Holiday", "Holiday"]

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Average sales comparison
        axes[0].bar(
            ["Non-Holiday", "Holiday"], holiday_impact["mean"], color=["lightblue", "salmon"]
        )
        axes[0].set_title(
            "Average Sales: Holidays vs Non-Holidays", fontsize=14, fontweight="bold"
        )
        axes[0].set_ylabel("Average Sales")

        # Monthly holiday distribution
        self.holiday_df["month"] = self.holiday_df["date"].dt.month
        monthly_holidays = self.holiday_df["month"].value_counts().sort_index()
        axes[1].bar(monthly_holidays.index, monthly_holidays.values, color="lightgreen")
        axes[1].set_title("Number of Holidays by Month", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Month")
        axes[1].set_ylabel("Number of Holidays")
        axes[1].set_xticks(range(1, 13))

        plt.tight_layout()
        plt.savefig(self.figures_dir / "holiday_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_transactions_analysis(self):
        """Plot transactions analysis"""
        logger.info("Creating transactions analysis plots...")

        # Merge transactions with sales data
        daily_store_sales = (
            self.train_df.groupby(["date", "store_nbr"])["sales"].sum().reset_index()
        )
        sales_transactions = daily_store_sales.merge(
            self.transactions_df, on=["date", "store_nbr"], how="inner"
        )

        # Calculate sales per transaction
        sales_transactions["sales_per_transaction"] = (
            sales_transactions["sales"] / sales_transactions["transactions"]
        )
        sales_transactions["sales_per_transaction"] = sales_transactions[
            "sales_per_transaction"
        ].replace([np.inf, -np.inf], np.nan)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Sales vs Transactions scatter plot
        sample_data = sales_transactions.sample(n=min(10000, len(sales_transactions)))
        axes[0, 0].scatter(sample_data["transactions"], sample_data["sales"], alpha=0.5)
        axes[0, 0].set_title("Sales vs Transactions (Sample)", fontsize=14, fontweight="bold")
        axes[0, 0].set_xlabel("Number of Transactions")
        axes[0, 0].set_ylabel("Total Sales")

        # Sales per transaction distribution
        valid_spt = sales_transactions["sales_per_transaction"].dropna()
        valid_spt = valid_spt[(valid_spt > 0) & (valid_spt < valid_spt.quantile(0.95))]
        axes[0, 1].hist(valid_spt, bins=50, alpha=0.7, color="green", edgecolor="black")
        axes[0, 1].set_title(
            "Distribution of Sales per Transaction", fontsize=14, fontweight="bold"
        )
        axes[0, 1].set_xlabel("Sales per Transaction ($)")
        axes[0, 1].set_ylabel("Frequency")

        # Monthly transaction trends
        monthly_transactions = self.transactions_df.copy()
        monthly_transactions["month"] = monthly_transactions["date"].dt.month
        monthly_trans_avg = monthly_transactions.groupby("month")["transactions"].mean()
        axes[1, 0].plot(monthly_trans_avg.index, monthly_trans_avg.values, marker="o", linewidth=2)
        axes[1, 0].set_title("Average Transactions by Month", fontsize=14, fontweight="bold")
        axes[1, 0].set_xlabel("Month")
        axes[1, 0].set_ylabel("Average Transactions")
        axes[1, 0].set_xticks(range(1, 13))
        axes[1, 0].grid(True, alpha=0.3)

        # Day of week transaction patterns
        self.transactions_df["dayofweek"] = self.transactions_df["date"].dt.dayofweek
        dow_transactions = self.transactions_df.groupby("dayofweek")["transactions"].mean()
        axes[1, 1].bar(range(len(dow_transactions)), dow_transactions.values, color="orange")
        axes[1, 1].set_title("Average Transactions by Day of Week", fontsize=14, fontweight="bold")
        axes[1, 1].set_xlabel("Day of Week")
        axes[1, 1].set_ylabel("Average Transactions")
        axes[1, 1].set_xticks(range(7))
        axes[1, 1].set_xticklabels(self.day_names, rotation=45)

        plt.tight_layout()
        plt.savefig(self.figures_dir / "transactions_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_correlation_matrix(self):
        """Plot correlation matrix"""
        logger.info("Creating correlation matrix plot...")

        # Create correlation dataset
        corr_data = self.train_df[
            ["sales", "onpromotion", "year", "month", "dayofweek", "is_holiday"]
        ].copy()
        correlation_matrix = corr_data.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )
        plt.title("Correlation Matrix of Key Variables", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.figures_dir / "correlation_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()

    def generate_all_plots(self):
        """Generate all EDA plots"""
        logger.info("Starting comprehensive EDA plot generation...")

        self.prepare_data()

        # Generate all plots
        self.plot_sales_distribution()
        self.plot_temporal_patterns()
        self.plot_sales_by_factors()
        self.plot_store_analysis()
        self.plot_family_analysis()
        self.plot_oil_analysis()
        self.plot_holiday_analysis()
        self.plot_transactions_analysis()
        self.plot_correlation_matrix()

        logger.success(f"All EDA plots generated successfully in {self.figures_dir}")


if __name__ == "__main__":
    """Generate comprehensive EDA plots from store sales data"""

    try:
        # Initialize plotter
        plotter = EDAPlotter(FIGURES_DIR)

        # Load data
        plotter.load_data(RAW_DATA_DIR)

        # Generate all plots
        plotter.generate_all_plots()

        logger.success("EDA plot generation complete!")

    except Exception as e:
        logger.error(f"Error generating plots: {e}")
        raise
