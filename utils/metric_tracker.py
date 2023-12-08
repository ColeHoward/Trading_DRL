from collections import defaultdict
import pandas as pd



class MetricTracker:
    def __init__(self, start_date, end_date, risk_free_rate=.04):
        self.train_dates = (start_date, end_date)
        self.metrics = ['price', 'portfolio_value', 'cash', 'shares_value', 'position', 'reward', 'sharpe_ratio']
        self.train_metrics = {metric: defaultdict(list) for metric in self.metrics}
        self.test_metrics = {metric: defaultdict(list) for metric in self.metrics}
        self.num_train_epochs = 1
        self.num_test_epochs = 1
        self.rfr = risk_free_rate

    def push(self, price, portfolio_value, cash, shares_value, position, reward, epoch, is_train):

        if is_train:
            self.num_train_epochs = max(self.num_train_epochs, epoch+1)
            self.train_metrics['price'][epoch].append(price)
            self.train_metrics['portfolio_value'][epoch].append(portfolio_value)
            self.train_metrics['cash'][epoch].append(cash)
            self.train_metrics['shares_value'][epoch].append(shares_value)
            self.train_metrics['position'][epoch].append(position)
            self.train_metrics['reward'][epoch].append(reward)
            if len(self.train_metrics['portfolio_value'][epoch]) > 1:
                sharpe_ratio = self.calc_sharpe_ratio(epoch, is_train)
            else:
                sharpe_ratio = 0
            self.train_metrics['sharpe_ratio'][epoch].append(sharpe_ratio)
        else:
            self.num_test_epochs = max(self.num_test_epochs, epoch+1)
            self.test_metrics['price'][epoch].append(price)
            self.test_metrics['portfolio_value'][epoch].append(portfolio_value)
            self.test_metrics['cash'][epoch].append(cash)
            self.test_metrics['shares_value'][epoch].append(shares_value)
            self.test_metrics['position'][epoch].append(position)
            self.test_metrics['reward'][epoch].append(reward)
            if len(self.test_metrics['portfolio_value'][epoch]) > 1:
                sharpe_ratio = self.calc_sharpe_ratio(epoch, is_train)
            else:
                sharpe_ratio = 0
            self.test_metrics['sharpe_ratio'][epoch].append(sharpe_ratio)


    def get_metrics_by_epoch(self, epoch, is_training) -> pd.DataFrame:
        res = []
        for metric in self.metrics:
            if is_training:
                for val in self.train_metrics[metric][epoch]:
                    res.append(val)
            else:
                for val in self.test_metrics[metric][epoch]:
                    res.append(val)

        return pd.DataFrame(res, columns=self.metrics)


    def calc_sharpe_ratio(self, epoch, is_training, start=0, end=-1) -> float:
        # if time, optimize by storing intermediate returns in class
        intermediate_returns = []
        if is_training:
            for i in range(start, len(self.train_metrics['portfolio_value'][epoch]) - 1):
                pv_start = self.train_metrics['portfolio_value'][epoch][i]
                pv_End = self.train_metrics['portfolio_value'][epoch][i + 1]
                intermediate_returns.append((pv_End - pv_start) / pv_start)

        else:
            for i in range(start, len(self.test_metrics['portfolio_value'][epoch]) - 1):
                pv_start = self.test_metrics['portfolio_value'][epoch][i]
                pv_End = self.test_metrics['portfolio_value'][epoch][i + 1]
                intermediate_returns.append((pv_End - pv_start) / pv_start)

        if len(intermediate_returns) == 0:
            return float('nan')  # Handle empty list case

        mean = sum(intermediate_returns) / len(intermediate_returns)
        variance = sum((ret - mean) ** 2 for ret in intermediate_returns)

        # Use len(intermediate_returns) - 1
        sd = (variance / (len(intermediate_returns) - 1)) ** 0.5 if len(intermediate_returns) > 1 else 0

        if sd == 0:
            return float('nan')

        sharpe = (mean - self.rfr) / sd

        return sharpe


    def write_metrics(self, csv_filename, is_training):
        if is_training:
            with open(csv_filename, 'w') as f:
                f.write(','.join(['epoch'] + self.metrics) + '\n')
                for epoch in range(self.num_train_epochs):
                    for i in range(len(self.train_metrics['portfolio_value'][epoch])):
                        row = [str(epoch)]
                        for metric in self.metrics:
                            row.append(str(self.train_metrics[metric][epoch][i]))
                        f.write(','.join(row) + '\n')
        else:
            with open(csv_filename, 'w') as f:
                f.write(','.join(['epoch'] + self.metrics) + '\n')
                for epoch in range(self.num_test_epochs):
                    for i in range(len(self.test_metrics['portfolio_value'][epoch])):
                        row = [str(epoch)]
                        for metric in self.metrics:
                            row.append(str(self.test_metrics[metric][epoch][i]))
                        f.write(','.join(row) + '\n')



