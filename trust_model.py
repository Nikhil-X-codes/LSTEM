import pandas as pd
import numpy as np
from typing import Dict, Tuple


class TrustScoreCalculator:

    def __init__(self, w_psr: float = 0.30, w_consistency: float = 0.35, 
                 w_compliance: float = 0.20, w_stability: float = 0.15):
        self.w_psr = w_psr
        self.w_consistency = w_consistency
        self.w_compliance = w_compliance
        self.w_stability = w_stability
        
        total = w_psr + w_consistency + w_compliance + w_stability
        if abs(total - 1.0) > 0.001:
            self.w_psr /= total
            self.w_consistency /= total
            self.w_compliance /= total
            self.w_stability /= total
    
    def calculate_packet_success_rate(self, device_df: pd.DataFrame) -> float:
        if len(device_df) == 0:
            return 0.0
        
        normal_characteristics = 0
        
        size_normal = (device_df['pck_size'] >= 60) & (device_df['pck_size'] <= 1600)
        normal_characteristics += size_normal.sum()
        
        standard_protocols = device_df['IP_proto'].isin([6, 17, 1])
        normal_characteristics += standard_protocols.sum()
        
        normal_ihl = device_df['IP_ihl'].isin([5, 6])
        normal_characteristics += normal_ihl.sum()
        
        normal_frag = device_df['IP_frag'] == 0
        normal_characteristics += normal_frag.sum()
        
        psr = normal_characteristics / (len(device_df) * 4)
        return np.clip(psr, 0.0, 1.0)
    
    def calculate_behavior_consistency(self, device_df: pd.DataFrame) -> float:
        if len(device_df) < 2:
            return 0.5
        
        consistency_scores = []
        
        if len(device_df['IP_proto'].unique()) <= 2:
            consistency_scores.append(0.9)
        elif len(device_df['IP_proto'].unique()) <= 4:
            consistency_scores.append(0.6)
        else:
            consistency_scores.append(0.3)
        
        pck_cv = device_df['pck_size'].std() / (device_df['pck_size'].mean() + 1e-6)
        pck_consistency = np.exp(-pck_cv / 2.0)
        consistency_scores.append(np.clip(pck_consistency, 0.0, 1.0))
        
        tcp_profile_consistency = 1.0 - (device_df[['TCP_ACK', 'TCP_FIN']].std().mean() / 0.5)
        consistency_scores.append(np.clip(tcp_profile_consistency, 0.0, 1.0))
        
        flag_entropy = device_df['IP_flags'].nunique() / 4.0
        flag_consistency = 1.0 - (flag_entropy * 0.3)
        consistency_scores.append(np.clip(flag_consistency, 0.0, 1.0))
        
        bc = np.mean(consistency_scores)
        return np.clip(bc, 0.0, 1.0)
    
    def calculate_protocol_compliance(self, device_df: pd.DataFrame) -> float:
        if len(device_df) == 0:
            return 0.5
        
        compliance_factors = []
        
        ihl_normal = (device_df['IP_ihl'] == 5).sum() / len(device_df)
        compliance_factors.append(np.clip(ihl_normal + 0.2, 0.0, 1.0))
        
        ack_rate = device_df['TCP_ACK'].mean()
        fin_rate = device_df['TCP_FIN'].mean()
        flag_balance = 1.0 - abs(ack_rate - 0.5) - abs(fin_rate - 0.2)
        compliance_factors.append(np.clip(flag_balance, 0.0, 1.0))
        
        frag_rate = device_df['IP_frag'].mean()
        frag_compliance = np.exp(-frag_rate * 5.0)
        compliance_factors.append(np.clip(frag_compliance, 0.0, 1.0))
        
        normal_flags = (device_df['IP_flags'] == 0).sum() / len(device_df)
        compliance_factors.append(np.clip(normal_flags + 0.3, 0.0, 1.0))
        
        pc = np.mean(compliance_factors)
        return np.clip(pc, 0.0, 1.0)
    
    def calculate_temporal_stability(self, device_df: pd.DataFrame) -> float:
        if len(device_df) < 3:
            return 0.5
        
        df_sorted = device_df.sort_values('ts').reset_index(drop=True)
        
        stability_metrics = []
        
        ts_diff = df_sorted['ts'].diff().dropna()
        if len(ts_diff) > 0:
            ts_cv = ts_diff.std() / (ts_diff.mean() + 1e-6)
            ts_stability = np.exp(-ts_cv / 3.0)
            stability_metrics.append(np.clip(ts_stability, 0.0, 1.0))
        
        if len(df_sorted) >= 4:
            chunk_size = max(2, len(df_sorted) // 4)
            size_means = [df_sorted['pck_size'].iloc[i:i+chunk_size].mean() 
                          for i in range(0, len(df_sorted), chunk_size)]
            size_chunk_cv = np.std(size_means) / (np.mean(size_means) + 1e-6)
            size_stability = np.exp(-size_chunk_cv)
            stability_metrics.append(np.clip(size_stability, 0.0, 1.0))
        
        protocol_variance = df_sorted['IP_proto'].std() / (df_sorted['IP_proto'].mean() + 1e-6 + 1)
        protocol_stability = np.exp(-protocol_variance)
        stability_metrics.append(np.clip(protocol_stability, 0.0, 1.0))
        
        ts = np.mean(stability_metrics) if stability_metrics else 0.5
        return np.clip(ts, 0.0, 1.0)
    
    def compute_trust_score(self, device_df: pd.DataFrame) -> float:
        psr = self.calculate_packet_success_rate(device_df)
        consistency = self.calculate_behavior_consistency(device_df)
        compliance = self.calculate_protocol_compliance(device_df)
        stability = self.calculate_temporal_stability(device_df)
        
        trust_score = (self.w_psr * psr + 
                      self.w_consistency * consistency + 
                      self.w_compliance * compliance + 
                      self.w_stability * stability)
        
        return np.clip(trust_score, 0.0, 1.0)
    
    def compute_batch_trust_scores(self, df: pd.DataFrame, 
                                   groupby_col: str = 'IP_src') -> pd.Series:
        trust_scores = {}
        
        for device_id, device_df in df.groupby(groupby_col):
            trust_score = self.compute_trust_score(device_df)
            trust_scores[device_id] = trust_score
        
        return pd.Series(trust_scores)


def add_trust_scores_to_dataframe(df: pd.DataFrame, 
                                   w_psr: float = 0.30,
                                   w_consistency: float = 0.35,
                                   w_compliance: float = 0.20,
                                   w_stability: float = 0.15,
                                   groupby_col: str = 'IP_src') -> pd.DataFrame:
    calculator = TrustScoreCalculator(w_psr, w_consistency, w_compliance, w_stability)
    trust_scores = calculator.compute_batch_trust_scores(df, groupby_col)
    
    result_df = df.copy()
    result_df['trust_score'] = result_df[groupby_col].map(trust_scores)
    
    return result_df
