from datetime import datetime
from allo_optim.optimizer.wikipedia import allocate_wikipedia
from allo_optim.config.stock_universe import list_of_dax_stocks

if __name__ == "__main__":
    stocks = list_of_dax_stocks()
    time_today = datetime(2024, 6, 1)

    result = allocate_wikipedia(stocks, time_today)
    print(f"Allocation result: {result}")

    # Log performance report
    print("Logging performance report...")
    print("Done.")
