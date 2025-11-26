"""Pipeline script to scrape health data from WebMD."""
import argparse
from pathlib import Path

from services.scraper_service import ScraperService


def main():
    parser = argparse.ArgumentParser(description="Scrape health data from WebMD")
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/patient_education.parquet",
        help="Output path for scraped data (parquet format)",
    )
    parser.add_argument(
        "--max-topics",
        type=int,
        default=None,
        help="Maximum number of topics to scrape (None for all)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between requests in seconds (default: 1.0)",
    )
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    
    print("[INFO] Starting WebMD scraping pipeline...")
    print(f"[INFO] Output path: {output_path}")
    print(f"[INFO] Max topics: {args.max_topics or 'all'}")
    print(f"[INFO] Request delay: {args.delay}s")
    
    # Initialize scraper
    scraper = ScraperService(delay=args.delay)
    
    # Scrape all topics
    scraped_data = scraper.scrape_all_topics(max_topics=args.max_topics)
    
    if not scraped_data:
        print("[ERROR] No data was scraped. Exiting.")
        return 1
    
    # Save to parquet
    df = scraper.save_to_dataframe(scraped_data, output_path)
    
    print(f"[INFO] Scraping pipeline completed successfully!")
    print(f"[INFO] Total records: {len(df)}")
    print(f"[INFO] Saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())

