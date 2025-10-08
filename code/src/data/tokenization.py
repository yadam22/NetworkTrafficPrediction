# Network Traffic Tokenization System
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Set
import logging
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import pickle
import json
from collections import defaultdict, Counter
import ipaddress
import hashlib
from dataclasses import dataclass
from enum import Enum
import os
import sys
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TokenType(Enum):
    """Token type categories for network traffic"""
    SPECIAL = "special"
    NETWORK = "network"
    TEMPORAL = "temporal"
    BEHAVIORAL = "behavioral"
    STATISTICAL = "statistical"
    PROTOCOL = "protocol"
    FLOW = "flow"
    ANOMALY = "anomaly"
    CONTEXTUAL = "contextual"

@dataclass
class TokenMetadata:
    """Metadata for each token"""
    token: str
    type: TokenType
    feature: str
    value_range: Optional[Tuple[float, float]] = None
    frequency: int = 0
    importance_score: float = 0.0

class NetworkTokenizer:
    """
    Advanced tokenizer for network traffic data with hierarchical, multi-scale,
    and network-aware tokenization strategies
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vocab_size = config.get('vocab_size', 50000)
        self.max_sequence_length = config.get('max_sequence_length', 1024)
        
        # Enhanced components
        self.discretizers = {}
        self.encoders = {}
        self.embedders = {}
        self.clusterers = {}
        
        # Vocabulary management
        self.vocabulary = {}
        self.token_to_id = {}
        self.id_to_token = {}
        self.token_metadata = {}
        
        # Special tokens with network-specific additions
        self.special_tokens = {
            'PAD': 0, 'UNK': 1, 'CLS': 2, 'SEP': 3, 'MASK': 4,
            'FLOW_START': 5, 'FLOW_END': 6, 'ANOMALY': 7,
            'BURST_START': 8, 'BURST_END': 9, 'SCAN': 10,
            'DDoS': 11, 'NORMAL': 12, 'TIME_SEP': 13
        }
        self.next_token_id = len(self.special_tokens)
        
        # Network-specific configurations
        self.protocol_hierarchy = self._build_protocol_hierarchy()
        self.port_categories = self._build_port_categories()
        self.temporal_scales = ['microsec', 'millisec', 'second', 'minute', 'hour', 'day']
        
        self.is_fitted = False
        
    def _build_protocol_hierarchy(self) -> Dict[str, List[str]]:
        return {
            'transport': ['TCP', 'UDP', 'ICMP', 'SCTP'],
            'application': {
                'TCP': ['HTTP', 'HTTPS', 'SSH', 'FTP', 'SMTP', 'POP3', 'IMAP'],
                'UDP': ['DNS', 'DHCP', 'NTP', 'SNMP', 'RTP', 'SIP']
            },
            'security': ['IPSec', 'TLS', 'SSH', 'VPN'],
            'streaming': ['RTP', 'RTSP', 'HLS', 'DASH'],
            'p2p': ['BitTorrent', 'eDonkey', 'Gnutella']
        }
    
    def _build_port_categories(self) -> Dict[str, Tuple[int, int]]:
        return {
            'well_known': (0, 1023),
            'registered': (1024, 49151),
            'dynamic': (49152, 65535),
            'http_alts': [8080, 8081, 8090, 8443],
            'database': [3306, 5432, 27017, 6379, 9200],
            'monitoring': [161, 162, 9090, 9100],
            'vpn': [500, 1194, 1701, 1723, 4500]
        }
    
    def _create_hierarchical_ip_tokens(self, ip_addresses: pd.Series) -> List[List[str]]:
        hierarchical_tokens = []
        
        for ip_str in ip_addresses:
            if pd.isna(ip_str) or ip_str == 'unknown':
                hierarchical_tokens.append(['IP_UNK'])
                continue
                
            try:
                ip = ipaddress.ip_address(ip_str)
                tokens = []
                
                if isinstance(ip, ipaddress.IPv4Address):
                    # Network class
                    first_octet = int(str(ip).split('.')[0])
                    if first_octet < 128:
                        tokens.append('IP_CLASS_A')
                    elif first_octet < 192:
                        tokens.append('IP_CLASS_B')
                    elif first_octet < 224:
                        tokens.append('IP_CLASS_C')
                    else:
                        tokens.append('IP_CLASS_OTHER')
                    
                    # Private/Public
                    if ip.is_private:
                        tokens.append('IP_PRIVATE')
                    else:
                        tokens.append('IP_PUBLIC')
                    
                    # Special addresses
                    if ip.is_multicast:
                        tokens.append('IP_MULTICAST')
                    elif ip.is_loopback:
                        tokens.append('IP_LOOPBACK')
                    
                    # Subnet tokens (anonymized)
                    subnet_hash = hashlib.md5(str(ip).encode()).hexdigest()[:8]
                    tokens.append(f'SUBNET_{subnet_hash[:4]}')
                    
                else:  # IPv6
                    tokens.append('IP_V6')
                    if ip.is_private:
                        tokens.append('IP6_PRIVATE')
                    if ip.is_multicast:
                        tokens.append('IP6_MULTICAST')
                
                hierarchical_tokens.append(tokens)
                
            except:
                hierarchical_tokens.append(['IP_INVALID'])
        
        return hierarchical_tokens
    
    def _create_port_tokens(self, ports: pd.Series) -> List[str]:
        port_tokens = []
        
        for port in ports:
            if pd.isna(port):
                port_tokens.append('PORT_UNK')
                continue
                
            port = int(port)
            tokens = []
            
            # Port range category
            if port <= 1023:
                tokens.append('PORT_WELL_KNOWN')
            elif port <= 49151:
                tokens.append('PORT_REGISTERED')
            else:
                tokens.append('PORT_DYNAMIC')
            
            # Specific service detection
            service_map = {
                20: 'SERVICE_FTP_DATA', 21: 'SERVICE_FTP',
                22: 'SERVICE_SSH', 23: 'SERVICE_TELNET',
                25: 'SERVICE_SMTP', 53: 'SERVICE_DNS',
                80: 'SERVICE_HTTP', 443: 'SERVICE_HTTPS',
                110: 'SERVICE_POP3', 143: 'SERVICE_IMAP',
                3306: 'SERVICE_MYSQL', 5432: 'SERVICE_POSTGRES',
                6379: 'SERVICE_REDIS', 27017: 'SERVICE_MONGODB'
            }
            
            if port in service_map:
                tokens.append(service_map[port])
            
            # Port patterns
            if port in self.port_categories['http_alts']:
                tokens.append('PORT_HTTP_ALT')
            elif port in self.port_categories['database']:
                tokens.append('PORT_DATABASE')
            elif port in self.port_categories['vpn']:
                tokens.append('PORT_VPN')
            
            # Combine tokens
            port_tokens.append('_'.join(tokens) if tokens else f'PORT_{port}')
        
        return port_tokens
    
    def _create_flow_pattern_tokens(self, df: pd.DataFrame) -> List[str]:
        flow_tokens = []
        
        for _, row in df.iterrows():
            tokens = []
            
            # Flow size patterns - adjusted for aggregated data
            if 'total_bytes_sum' in row:
                bytes_val = row['total_bytes_sum']
                if bytes_val < 10000:
                    tokens.append('FLOW_TINY')
                elif bytes_val < 100000:
                    tokens.append('FLOW_SMALL')
                elif bytes_val < 1000000:
                    tokens.append('FLOW_MEDIUM')
                elif bytes_val < 10000000:
                    tokens.append('FLOW_LARGE')
                else:
                    tokens.append('FLOW_HUGE')
            
            # Packet count patterns
            if 'packet_count_sum' in row:
                packet_count = row['packet_count_sum']
                if packet_count < 100:
                    tokens.append('PACKETS_LOW')
                elif packet_count < 1000:
                    tokens.append('PACKETS_MEDIUM')
                elif packet_count < 10000:
                    tokens.append('PACKETS_HIGH')
                else:
                    tokens.append('PACKETS_EXTREME')
            
            # Packet rate patterns
            if 'packet_rate_mean' in row:
                rate = row['packet_rate_mean']
                if rate < 10:
                    tokens.append('RATE_LOW')
                elif rate < 100:
                    tokens.append('RATE_MEDIUM')
                elif rate < 1000:
                    tokens.append('RATE_HIGH')
                else:
                    tokens.append('RATE_EXTREME')
            
            # Flow diversity
            if 'flow_id_nunique' in row:
                flow_count = row['flow_id_nunique']
                if flow_count < 10:
                    tokens.append('FLOWS_FEW')
                elif flow_count < 100:
                    tokens.append('FLOWS_MODERATE')
                elif flow_count < 1000:
                    tokens.append('FLOWS_MANY')
                else:
                    tokens.append('FLOWS_MASSIVE')
            
            flow_tokens.append('_'.join(tokens) if tokens else 'FLOW_NORMAL')
        
        return flow_tokens
    
    def _create_temporal_multiscale_tokens(self, timestamps: pd.Series) -> List[List[str]]:
        temporal_tokens = []
        
        for ts in timestamps:
            if pd.isna(ts):
                temporal_tokens.append(['TIME_UNK'])
                continue
            
            tokens = []
            
            # Multiple time scales
            microsec = ts.microsecond // 100000  # 0-9
            tokens.append(f'MICROSEC_{microsec}')
            
            # Second scale
            second = ts.second // 10  # 0-5
            tokens.append(f'SEC_BLOCK_{second}')
            
            # Minute scale
            minute_quarter = ts.minute // 15  # 0-3
            tokens.append(f'MINUTE_Q{minute_quarter}')
            
            # Hour scale with business context
            hour = ts.hour
            if 9 <= hour <= 17:
                tokens.append('HOUR_BUSINESS')
            elif 18 <= hour <= 23:
                tokens.append('HOUR_EVENING')
            elif 0 <= hour <= 5:
                tokens.append('HOUR_NIGHT')
            else:
                tokens.append('HOUR_MORNING')
            
            # Day patterns
            dow = ts.dayofweek
            if dow < 5:
                tokens.append('DAY_WEEKDAY')
            else:
                tokens.append('DAY_WEEKEND')
            
            week_of_month = (ts.day - 1) // 7
            tokens.append(f'WEEK_{week_of_month}')
            
            month = ts.month
            if month in [12, 1, 2]:
                tokens.append('SEASON_WINTER')
            elif month in [3, 4, 5]:
                tokens.append('SEASON_SPRING')
            elif month in [6, 7, 8]:
                tokens.append('SEASON_SUMMER')
            else:
                tokens.append('SEASON_FALL')
            
            temporal_tokens.append(tokens)
        
        return temporal_tokens
    
    def _create_behavioral_tokens(self, df: pd.DataFrame) -> List[str]:
        behavioral_tokens = []
        
        for _, row in df.iterrows():
            tokens = []
    
            # IP diversity patterns
            if 'src_ip_nunique' in row and 'dst_ip_nunique' in row:
                src_diversity = row['src_ip_nunique']
                dst_diversity = row['dst_ip_nunique']
                
                if src_diversity < 5 and dst_diversity < 5:
                    tokens.append('DIVERSITY_LOW')
                elif src_diversity < 50 or dst_diversity < 50:
                    tokens.append('DIVERSITY_MEDIUM')
                else:
                    tokens.append('DIVERSITY_HIGH')
                
                # Check for scanning behavior
                if src_diversity < 5 and dst_diversity > 100:
                    tokens.append('BEHAVIOR_SCAN_OUT')
                elif src_diversity > 100 and dst_diversity < 5:
                    tokens.append('BEHAVIOR_SCAN_IN')
            
            # Data transfer patterns based on packet size
            if 'mean_packet_size_mean' in row:
                avg_size = row['mean_packet_size_mean']
                if avg_size < 100:
                    tokens.append('TRANSFER_CONTROL')
                elif avg_size < 500:
                    tokens.append('TRANSFER_MIXED')
                elif avg_size < 1400:
                    tokens.append('TRANSFER_BULK')
                else:
                    tokens.append('TRANSFER_JUMBO')
            
            # Rate patterns for anomaly detection
            if 'packet_rate_mean' in row and 'byte_rate_mean' in row:
                packet_rate = row['packet_rate_mean']
                byte_rate = row['byte_rate_mean']
                
                if packet_rate > 1000 and byte_rate < 100000:
                    tokens.append('BEHAVIOR_MANY_SMALL')
                elif packet_rate < 100 and byte_rate > 1000000:
                    tokens.append('BEHAVIOR_FEW_LARGE')
            
            behavioral_tokens.append('_'.join(tokens) if tokens else 'BEHAVIOR_NORMAL')
        
        return behavioral_tokens
    
    def _create_statistical_tokens(self, feature_name: str, values: np.ndarray, 
                                  n_clusters: int = 20) -> List[str]:
        if len(np.unique(values)) < 2:
            return [f"{feature_name}_CONST" for _ in values]
        
        values_clean = values[~np.isnan(values)]
        if len(values_clean) < n_clusters:
            n_clusters = max(2, len(np.unique(values_clean)))
        
        scaler = StandardScaler()
        values_scaled = scaler.fit_transform(values_clean.reshape(-1, 1))
        
        dbscan = DBSCAN(eps=2.0, min_samples=5)
        outlier_labels = dbscan.fit_predict(values_scaled)
        
        non_outliers = values_scaled[outlier_labels != -1]
        if len(non_outliers) > n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(non_outliers)
            
            # Create tokens
            tokens = []
            outlier_idx = 0
            regular_idx = 0
            
            for i, val in enumerate(values):
                if np.isnan(val):
                    tokens.append(f"{feature_name}_NAN")
                elif outlier_labels[outlier_idx] == -1:
                    tokens.append(f"{feature_name}_OUTLIER")
                    outlier_idx += 1
                else:
                    cluster = kmeans.predict(values_scaled[regular_idx].reshape(1, -1))[0]
                    tokens.append(f"{feature_name}_C{cluster}")
                    outlier_idx += 1
                    regular_idx += 1
            
            self.clusterers[feature_name] = (scaler, dbscan, kmeans)
        else:
            # Simple binning for small datasets
            discretizer = KBinsDiscretizer(n_bins=min(n_clusters, len(non_outliers)), 
                                          encode='ordinal', strategy='quantile')
            bins = discretizer.fit_transform(values.reshape(-1, 1)).flatten()
            tokens = [f"{feature_name}_B{int(b)}" for b in bins]
            self.discretizers[feature_name] = discretizer
        
        return tokens
    
    def _create_anomaly_score_tokens(self, df: pd.DataFrame) -> List[str]:
        anomaly_tokens = []
        
        for _, row in df.iterrows():
            anomaly_score = 0
            
            # Check for unusual patterns - adjusted for aggregated data
            if 'packet_rate_mean' in row and row['packet_rate_mean'] > 1000:
                anomaly_score += 2
            
            if 'packet_count_sum' in row and 'flow_id_nunique' in row:
                # High packet count with low flow diversity might indicate attack
                if row['packet_count_sum'] > 10000 and row['flow_id_nunique'] < 10:
                    anomaly_score += 3
            
            if 'src_ip_nunique' in row and row['src_ip_nunique'] > 1000:
                anomaly_score += 2
            
            if 'byte_rate_mean' in row and row['byte_rate_mean'] > 10000000:
                anomaly_score += 1
            
            if anomaly_score == 0:
                anomaly_tokens.append('ANOMALY_NONE')
            elif anomaly_score < 3:
                anomaly_tokens.append('ANOMALY_LOW')
            elif anomaly_score < 6:
                anomaly_tokens.append('ANOMALY_MEDIUM')
            else:
                anomaly_tokens.append('ANOMALY_HIGH')
        
        return anomaly_tokens
    
    def _build_hierarchical_vocabulary(self, all_tokens: List[Union[str, List[str]]]) -> None:
        flat_tokens = []
        token_hierarchy = defaultdict(list)
        
        for token_group in all_tokens:
            if isinstance(token_group, list):
                for i, token in enumerate(token_group):
                    flat_tokens.append(token)
                    if i > 0:
                        token_hierarchy[token_group[i-1]].append(token)
            else:
                flat_tokens.append(token_group)
        
        token_counts = Counter(flat_tokens)
        
        token_scores = {}
        for token, count in token_counts.items():
            score = np.log(count + 1)
            
            # Boost scores for important token types
            if token.startswith(('ANOMALY_', 'BEHAVIOR_', 'ATTACK_')):
                score *= 2.0
            elif token.startswith(('SERVICE_', 'PROTOCOL_')):
                score *= 1.5
            elif token.startswith(('FLOW_', 'RATE_')):
                score *= 1.3
            
            token_scores[token] = score
        
        # Select vocabulary based on scores
        sorted_tokens = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)
        available_vocab_size = self.vocab_size - len(self.special_tokens)
        selected_tokens = sorted_tokens[:available_vocab_size]
        
        # build token mappings
        self.token_to_id = self.special_tokens.copy()
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}
        
        for token, score in selected_tokens:
            if token not in self.token_to_id:
                self.token_to_id[token] = self.next_token_id
                self.id_to_token[self.next_token_id] = token
                
                # store metadata
                self.token_metadata[token] = TokenMetadata(
                    token=token,
                    type=self._get_token_type(token),
                    feature=self._extract_feature_name(token),
                    frequency=token_counts[token],
                    importance_score=score
                )
                
                self.next_token_id += 1
        
        # store hierarchy information
        self.token_hierarchy = dict(token_hierarchy)
        
        logger.info(f"Built hierarchical vocabulary with {len(self.token_to_id)} tokens")
    
    def _get_token_type(self, token: str) -> TokenType:
        prefixes = {
            'IP_': TokenType.NETWORK,
            'PORT_': TokenType.NETWORK,
            'SUBNET_': TokenType.NETWORK,
            'SERVICE_': TokenType.PROTOCOL,
            'FLOW_': TokenType.FLOW,
            'FLOWS_': TokenType.FLOW,
            'PACKETS_': TokenType.FLOW,
            'TIME_': TokenType.TEMPORAL,
            'MICROSEC_': TokenType.TEMPORAL,
            'SEC_': TokenType.TEMPORAL,
            'MINUTE_': TokenType.TEMPORAL,
            'HOUR_': TokenType.TEMPORAL,
            'DAY_': TokenType.TEMPORAL,
            'WEEK_': TokenType.TEMPORAL,
            'SEASON_': TokenType.TEMPORAL,
            'BEHAVIOR_': TokenType.BEHAVIORAL,
            'DIVERSITY_': TokenType.BEHAVIORAL,
            'TRANSFER_': TokenType.BEHAVIORAL,
            'ANOMALY_': TokenType.ANOMALY,
            'RATE_': TokenType.STATISTICAL,
            'DURATION_': TokenType.STATISTICAL
        }
        
        for prefix, token_type in prefixes.items():
            if token.startswith(prefix):
                return token_type
        
        return TokenType.CONTEXTUAL
    
    def _extract_feature_name(self, token: str) -> str:
        """Extract feature name from token"""
        parts = token.split('_')
        if len(parts) > 1:
            return parts[0].lower()
        return 'unknown'
    
    def fit(self, df: pd.DataFrame) -> 'NetworkTokenizer':
        """Fit the tokenizer on training data"""
        logger.info("Fitting NetworkTokenizer")
        
        all_tokens = []
        
        #Network layer tokens (if available in aggregated data)
        if 'src_ip_nunique' in df.columns:
            # Create tokens based on IP diversity
            diversity_tokens = []
            for val in df['src_ip_nunique']:
                if val < 10:
                    diversity_tokens.append('IP_DIVERSITY_LOW')
                elif val < 100:
                    diversity_tokens.append('IP_DIVERSITY_MEDIUM')
                else:
                    diversity_tokens.append('IP_DIVERSITY_HIGH')
            all_tokens.extend(diversity_tokens)
        
        # Flow pattern tokens
        flow_tokens = self._create_flow_pattern_tokens(df)
        all_tokens.extend(flow_tokens)
        
        # Multi-scale temporal tokens
        if 'datetime' in df.columns:
            temporal_tokens = self._create_temporal_multiscale_tokens(df['datetime'])
            all_tokens.extend(temporal_tokens)
        
        # Behavioral tokens
        behavioral_tokens = self._create_behavioral_tokens(df)
        all_tokens.extend(behavioral_tokens)
        
        # Statistical tokens for key numerical features
        statistical_features = [
            'packet_count_sum', 'total_bytes_sum', 'packet_rate_mean',
            'byte_rate_mean', 'mean_packet_size_mean'
        ]
        
        for feature in statistical_features:
            if feature in df.columns:
                stat_tokens = self._create_statistical_tokens(feature, df[feature].values)
                all_tokens.extend(stat_tokens)
        
        #Anomaly tokens
        anomaly_tokens = self._create_anomaly_score_tokens(df)
        all_tokens.extend(anomaly_tokens)
        
        # Direction tokens (if available)
        if 'direction' in df.columns:
            direction_tokens = [f"DIR_{d.upper()}" for d in df['direction']]
            all_tokens.extend(direction_tokens)
        
        self._build_hierarchical_vocabulary(all_tokens)
        
        self.is_fitted = True
        logger.info("NetworkTokenizer fitted successfully")
        return self
    
    def transform(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before transform")
        
        logger.info(f"Tokenizing {len(df)} samples with tokenizer")
        
        tokenized_sequences = []
        token_types = []
        position_ids = []
        
        for idx, row in df.iterrows():
            sequence_tokens = []
            sequence_types = []
            
            # CLS token
            sequence_tokens.append(self.special_tokens['CLS'])
            sequence_types.append(TokenType.SPECIAL.value)
            
            # Network diversity tokens
            if 'src_ip_nunique' in df.columns:
                val = row['src_ip_nunique']
                if val < 10:
                    token = 'IP_DIVERSITY_LOW'
                elif val < 100:
                    token = 'IP_DIVERSITY_MEDIUM'
                else:
                    token = 'IP_DIVERSITY_HIGH'
                token_id = self.token_to_id.get(token, self.special_tokens['UNK'])
                sequence_tokens.append(token_id)
                sequence_types.append(TokenType.NETWORK.value)
            
            # TIME_SEP to separate network from temporal
            sequence_tokens.append(self.special_tokens['TIME_SEP'])
            sequence_types.append(TokenType.SPECIAL.value)
            
            # Temporal tokens
            if 'datetime' in df.columns:
                temporal_tokens = self._create_temporal_multiscale_tokens(pd.Series([row['datetime']]))[0]
                for token in temporal_tokens:
                    token_id = self.token_to_id.get(token, self.special_tokens['UNK'])
                    sequence_tokens.append(token_id)
                    sequence_types.append(TokenType.TEMPORAL.value)
            
            # Flow pattern token
            flow_token = self._create_flow_pattern_tokens(pd.DataFrame([row]))[0]
            token_id = self.token_to_id.get(flow_token, self.special_tokens['UNK'])
            sequence_tokens.append(token_id)
            sequence_types.append(TokenType.FLOW.value)
            
            # Behavioral token
            behavioral_token = self._create_behavioral_tokens(pd.DataFrame([row]))[0]
            token_id = self.token_to_id.get(behavioral_token, self.special_tokens['UNK'])
            sequence_tokens.append(token_id)
            sequence_types.append(TokenType.BEHAVIORAL.value)
            
            # Statistical tokens for key features
            for feature in ['packet_rate_mean', 'byte_rate_mean', 'mean_packet_size_mean']:
                if feature in row and pd.notna(row[feature]):
                    # Use fitted clusterer/discretizer
                    if feature in self.clusterers:
                        scaler, _, kmeans = self.clusterers[feature]
                        val_scaled = scaler.transform([[row[feature]]])
                        cluster = kmeans.predict(val_scaled)[0]
                        token = f"{feature}_C{cluster}"
                    elif feature in self.discretizers:
                        bin_id = self.discretizers[feature].transform([[row[feature]]])[0][0]
                        token = f"{feature}_B{int(bin_id)}"
                    else:
                        token = f"{feature}_UNK"
                    
                    token_id = self.token_to_id.get(token, self.special_tokens['UNK'])
                    sequence_tokens.append(token_id)
                    sequence_types.append(TokenType.STATISTICAL.value)
            
            # Anomaly token
            anomaly_token = self._create_anomaly_score_tokens(pd.DataFrame([row]))[0]
            token_id = self.token_to_id.get(anomaly_token, self.special_tokens['UNK'])
            sequence_tokens.append(token_id)
            sequence_types.append(TokenType.ANOMALY.value)
            
            # Direction token if available
            if 'direction' in row:
                dir_token = f"DIR_{row['direction'].upper()}"
                token_id = self.token_to_id.get(dir_token, self.special_tokens['UNK'])
                sequence_tokens.append(token_id)
                sequence_types.append(TokenType.CONTEXTUAL.value)
            
            # SEP token
            sequence_tokens.append(self.special_tokens['SEP'])
            sequence_types.append(TokenType.SPECIAL.value)
            
            if len(sequence_tokens) > self.max_sequence_length:
                sequence_tokens = sequence_tokens[:self.max_sequence_length]
                sequence_types = sequence_types[:self.max_sequence_length]
            
            #pad sequences
            padding_length = self.max_sequence_length - len(sequence_tokens)
            sequence_tokens.extend([self.special_tokens['PAD']] * padding_length)
            sequence_types.extend([TokenType.SPECIAL.value] * padding_length)
            
            tokenized_sequences.append(sequence_tokens)
            token_types.append(sequence_types)
            
            # Position IDs
            pos_ids = list(range(len(sequence_tokens)))
            position_ids.append(pos_ids)
        
        # Create attention masks
        attention_masks = []
        for seq in tokenized_sequences:
            mask = [1 if token_id != self.special_tokens['PAD'] else 0 for token_id in seq]
            attention_masks.append(mask)
        
        # Token type IDs (for different embedding tables)
        token_type_ids = []
        for types in token_types:
            type_ids = [self._token_type_to_id(t) for t in types]
            token_type_ids.append(type_ids)
        
        result = {
            'input_ids': np.array(tokenized_sequences),
            'attention_mask': np.array(attention_masks),
            'token_type_ids': np.array(token_type_ids),
            'position_ids': np.array(position_ids)
        }
        
        logger.info(f"Tokenization complete. Shape: {result['input_ids'].shape}")
        return result
    
    def fit_transform(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        self.fit(df)
        return self.transform(df)
    
    def decode(self, token_ids: Union[List[int], np.ndarray]) -> List[str]:
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        
        decoded_tokens = []
        for token_id in token_ids:
            token = self.id_to_token.get(token_id, 'UNK')
            decoded_tokens.append(token)
        
        return decoded_tokens
    
    def _token_type_to_id(self, token_type: str) -> int:
        type_map = {
            TokenType.SPECIAL.value: 0,
            TokenType.NETWORK.value: 1,
            TokenType.TEMPORAL.value: 2,
            TokenType.BEHAVIORAL.value: 3,
            TokenType.STATISTICAL.value: 4,
            TokenType.PROTOCOL.value: 5,
            TokenType.FLOW.value: 6,
            TokenType.ANOMALY.value: 7,
            TokenType.CONTEXTUAL.value: 8
        }
        return type_map.get(token_type, 0)
    
    def create_mlm_data(self, token_ids: np.ndarray, mask_prob: float = 0.15, whole_word_mask: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        masked_sequences = token_ids.copy()
        labels = np.full_like(token_ids, -100)  # -100 = ignore in loss
        
        for i in range(len(token_ids)):
            sequence_metadata = []
            for token_id in token_ids[i]:
                token = self.id_to_token.get(token_id, 'UNK')
                metadata = self.token_metadata.get(token)
                sequence_metadata.append(metadata)
            
            masked_positions = set()
            
            for j in range(len(token_ids[i])):
                if token_ids[i][j] in self.special_tokens.values():
                    continue
                
                if j in masked_positions:
                    continue
                
                if np.random.random() < mask_prob:
                    if whole_word_mask and sequence_metadata[j]:
                        token_type = sequence_metadata[j].type
                        feature = sequence_metadata[j].feature
                        for k in range(max(0, j-5), min(len(token_ids[i]), j+5)):
                            if (sequence_metadata[k] and 
                                sequence_metadata[k].type == token_type and
                                sequence_metadata[k].feature == feature):
                                masked_positions.add(k)
                    else:
                        masked_positions.add(j)
            
            # Apply masking
            for pos in masked_positions:
                labels[i][pos] = token_ids[i][pos]
                
                # 80% MASK, 10% random, 10% unchanged
                rand = np.random.random()
                if rand < 0.8:
                    masked_sequences[i][pos] = self.special_tokens['MASK']
                elif rand < 0.9:
                    # Random token from same type if possible
                    token_type = sequence_metadata[pos].type if sequence_metadata[pos] else None
                    candidates = [tid for tid, token in self.id_to_token.items() 
                                 if tid >= len(self.special_tokens) and
                                 self.token_metadata.get(token, TokenMetadata('', TokenType.CONTEXTUAL, '')).type == token_type]
                    if candidates:
                        masked_sequences[i][pos] = np.random.choice(candidates)
                    else:
                        masked_sequences[i][pos] = np.random.randint(len(self.special_tokens), len(self.token_to_id))
        
        return masked_sequences, labels
    
    def create_anomaly_detection_data(self, token_ids: np.ndarray, 
                                    anomaly_injection_prob: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        augmented_sequences = token_ids.copy()
        anomaly_labels = np.zeros(len(token_ids))
        
        for i in range(len(token_ids)):
            if np.random.random() < anomaly_injection_prob:
                # Inject anomaly
                anomaly_labels[i] = 1
                
                # Choose anomaly type
                anomaly_type = np.random.choice(['shuffle', 'repeat', 'inject', 'remove'])
                
                # Get non-special token positions
                valid_positions = [j for j in range(len(token_ids[i]))
                                 if token_ids[i][j] not in self.special_tokens.values()]
                
                if anomaly_type == 'shuffle' and len(valid_positions) > 5:
                    # Shuffle a subsequence
                    start = np.random.randint(0, len(valid_positions)-5)
                    indices = valid_positions[start:start+5]
                    np.random.shuffle(indices)
                    
                elif anomaly_type == 'repeat' and len(valid_positions) > 3:
                    #  Repeat a pattern
                    start = np.random.randint(0, len(valid_positions)-3)
                    pattern = [augmented_sequences[i][valid_positions[j]] 
                             for j in range(start, start+3)]
                    insert_pos = np.random.randint(start+3, len(valid_positions))
                    for k, token in enumerate(pattern):
                        if insert_pos + k < len(valid_positions):
                            augmented_sequences[i][valid_positions[insert_pos + k]] = token
                
                elif anomaly_type == 'inject':
                    # anomaly tokens
                    anomaly_tokens = [tid for tid, token in self.id_to_token.items()
                                    if 'ANOMALY_' in token or 'ATTACK_' in token]
                    if anomaly_tokens and valid_positions:
                        inject_pos = np.random.choice(valid_positions)
                        augmented_sequences[i][inject_pos] = np.random.choice(anomaly_tokens)
                
                elif anomaly_type == 'remove' and len(valid_positions) > 10:
                    # Remove some tokens 
                    remove_count = np.random.randint(1, 5)
                    remove_positions = np.random.choice(valid_positions, remove_count, replace=False)
                    for pos in remove_positions:
                        augmented_sequences[i][pos] = self.special_tokens['PAD']
        
        return augmented_sequences, anomaly_labels
    
    def get_token_importance(self, token_id: int) -> float:
        token = self.id_to_token.get(token_id, 'UNK')
        metadata = self.token_metadata.get(token)
        return metadata.importance_score if metadata else 0.0
    
    def get_vocabulary_stats(self) -> Dict[str, Any]:
        stats = {
            'total_tokens': len(self.token_to_id),
            'special_tokens': len(self.special_tokens),
            'token_types': {}
        }
        
        #Count tokens by type
        for token, metadata in self.token_metadata.items():
            token_type = metadata.type.value
            if token_type not in stats['token_types']:
                stats['token_types'][token_type] = 0
            stats['token_types'][token_type] += 1
        
        # top tokens by importance
        top_tokens = sorted(self.token_metadata.items(), 
                          key=lambda x: x[1].importance_score, 
                          reverse=True)[:20]
        stats['top_important_tokens'] = [(t, m.importance_score) for t, m in top_tokens]
        
        # token frequency distribution
        frequencies = [m.frequency for m in self.token_metadata.values()]
        stats['frequency_stats'] = {
            'mean': np.mean(frequencies),
            'std': np.std(frequencies),
            'min': np.min(frequencies),
            'max': np.max(frequencies),
            'median': np.median(frequencies)
        }
        
        return stats
    
    def save_tokenizer(self, filepath: str) -> None:
        logger.info(f"Saving tokenizer to {filepath}")
        
        tokenizer_data = {
            'config': self.config,
            'discretizers': self.discretizers,
            'clusterers': self.clusterers,
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token,
            'token_metadata': self.token_metadata,
            'special_tokens': self.special_tokens,
            'next_token_id': self.next_token_id,
            'token_hierarchy': getattr(self, 'token_hierarchy', {}),
            'protocol_hierarchy': self.protocol_hierarchy,
            'port_categories': self.port_categories,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(tokenizer_data, f)
        
        logger.info("Tokenizer saved successfully")
    
    def load_tokenizer(self, filepath: str) -> None:
        logger.info(f"Loading tokenizer from {filepath}")
        
        with open(filepath, 'rb') as f:
            tokenizer_data = pickle.load(f)
        
        self.config = tokenizer_data['config']
        self.discretizers = tokenizer_data['discretizers']
        self.clusterers = tokenizer_data['clusterers']
        self.token_to_id = tokenizer_data['token_to_id']
        self.id_to_token = tokenizer_data['id_to_token']
        self.token_metadata = tokenizer_data['token_metadata']
        self.special_tokens = tokenizer_data['special_tokens']
        self.next_token_id = tokenizer_data['next_token_id']
        self.token_hierarchy = tokenizer_data.get('token_hierarchy', {})
        self.protocol_hierarchy = tokenizer_data['protocol_hierarchy']
        self.port_categories = tokenizer_data['port_categories']
        self.is_fitted = tokenizer_data['is_fitted']
        
        logger.info("Tokenizer loaded successfully")


def main():
    PROCESSED_DATA_DIR = Path("code/data/processed_data")
    OUTPUT_DIR = Path("code/data/tokenized_data")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Tokenizer configuration
    tokenizer_config = {
        'vocab_size': 10000,  # adjusted for available data
        'max_sequence_length': 256,  # adjusted for memory efficiency
        'numerical_bins': 50,
        'n_clusters': 10,
        'enable_hierarchical': True,
        'enable_anomaly_detection': True
    }
    
    print(f"Start time: {datetime.now()}")
    
    try:
        # check for processed data
        print("\n1. CHECKING FOR PROCESSED DATA")
        print("-"*40)
        
        if not PROCESSED_DATA_DIR.exists():
            raise FileNotFoundError(f"Processed data directory not found at {PROCESSED_DATA_DIR}")
        
        print(f"✓ Found processed data directory: {PROCESSED_DATA_DIR}")
        
        # process data from each time window
        print("\n2. LOADING PROCESSED DATA FILES")
        print("-"*40)
        
        tokenized_results = {}
        # updated time windows to match preprocessing output
        time_windows = ['10s', '30s', '1min']
        
        for window in time_windows:
            window_dir = PROCESSED_DATA_DIR / f"{window}_window"
            
            if window_dir.exists():
                # load normalized data
                normalized_data_path = window_dir / "normalized_data.csv"
                if normalized_data_path.exists():
                    print(f"\nProcessing {window} window data...")
                    df = pd.read_csv(normalized_data_path)
                    
                    # cnvert datetime string back to datetime object
                    if 'datetime' in df.columns:
                        df['datetime'] = pd.to_datetime(df['datetime'])
                    
                    print(f"  Loaded {len(df)} samples")
                    print(f"  Columns: {len(df.columns)}")
                    
                    # show available columns for debugging
                    print(f"  Available columns: {df.columns.tolist()[:10]}...")
                    
                    # initialize and fit tokenizer
                    print(f"\n3. TOKENIZING {window.upper()} WINDOW DATA")
                    print("-"*40)
                    
                    # create tokenizer instance
                    tokenizer = NetworkTokenizer(tokenizer_config)
                    
                    # fit and transform in one step
                    print("  Fitting and transforming data...")
                    tokenized_data = tokenizer.fit_transform(df)
                    
                    # vocabulary statistics
                    vocab_stats = tokenizer.get_vocabulary_stats()
                    print(f"  Vocabulary size: {vocab_stats['total_tokens']:,}")
                    print(f"  Token types distribution:")
                    for token_type, count in vocab_stats['token_types'].items():
                        print(f"    - {token_type}: {count:,}")
                    
                    print(f"  Tokenized shape: {tokenized_data['input_ids'].shape}")
                    print(f"  Average sequence length: {np.mean(np.sum(tokenized_data['attention_mask'], axis=1)):.1f}")
                    
                    # create pre-training data
                    print("\n4. CREATING PRE-TRAINING DATA")
                    print("-"*40)
                    
                    # create MLM data
                    mlm_inputs, mlm_labels = tokenizer.create_mlm_data(
                        tokenized_data['input_ids'], 
                        mask_prob=0.15
                    )
                    print(f"  MLM data created")
                    print(f"    - Masked positions: {np.sum(mlm_labels != -100):,}")
                    
                    # create anomaly detection data
                    anomaly_inputs, anomaly_labels = tokenizer.create_anomaly_detection_data(
                        tokenized_data['input_ids'],
                        anomaly_injection_prob=0.1
                    )
                    print(f"  Anomaly detection data created")
                    print(f"    - Normal sequences: {np.sum(anomaly_labels == 0)}")
                    print(f"    - Anomalous sequences: {np.sum(anomaly_labels == 1)}")
                    
                    # save tokenized data
                    print("\n5. SAVING TOKENIZED DATA")
                    print("-"*40)
                    
                    # save tokenizer
                    tokenizer_path = OUTPUT_DIR / f"tokenizer_{window}.pkl"
                    tokenizer.save_tokenizer(str(tokenizer_path))
                    print(f"  Tokenizer saved to {tokenizer_path}")
                    
                    # save tokenized data
                    tokenized_data_path = OUTPUT_DIR / f"tokenized_data_{window}.npz"
                    np.savez_compressed(
                        tokenized_data_path,
                        input_ids=tokenized_data['input_ids'],
                        attention_mask=tokenized_data['attention_mask'],
                        token_type_ids=tokenized_data['token_type_ids'],
                        position_ids=tokenized_data['position_ids'],
                        mlm_inputs=mlm_inputs,
                        mlm_labels=mlm_labels,
                        anomaly_inputs=anomaly_inputs,
                        anomaly_labels=anomaly_labels
                    )
                    print(f"  Tokenized data saved to {tokenized_data_path}")
                    
                    # save vocabulary analysis
                    vocab_analysis = {
                        'vocab_stats': vocab_stats,
                        'unique_tokens_used': len(set(tokenized_data['input_ids'].flatten())),
                        'total_sequences': len(tokenized_data['input_ids']),
                        'avg_sequence_length': float(np.mean(np.sum(tokenized_data['attention_mask'], axis=1))),
                        'window': window
                    }
                    
                    # convert numpy types to native Python types for JSON
                    vocab_analysis['vocab_stats']['frequency_stats'] = {
                        k: float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v
                        for k, v in vocab_analysis['vocab_stats']['frequency_stats'].items()
                    }
                    vocab_analysis['vocab_stats']['top_important_tokens'] = [
                        (token, float(score)) for token, score in vocab_analysis['vocab_stats']['top_important_tokens']
                    ]
                    
                    vocab_analysis_path = OUTPUT_DIR / f"vocab_analysis_{window}.json"
                    with open(vocab_analysis_path, 'w') as f:
                        json.dump(vocab_analysis, f, indent=2, default=str)
                    print(f"  Vocabulary analysis saved to {vocab_analysis_path}")
                    
                    # Store results
                    tokenized_results[window] = {
                        'tokenizer': tokenizer,
                        'tokenized_data': tokenized_data,
                        'mlm_data': (mlm_inputs, mlm_labels),
                        'anomaly_data': (anomaly_inputs, anomaly_labels),
                        'stats': vocab_analysis
                    }
                    
                else:
                    print(f"  No normalized data found for {window} window")
            else:
                print(f" No data directory found for {window} window")
        
        if tokenized_results:
            print("\n" + "="*80)
            print("GENERATING VISUALIZATIONS")
            print("="*80)
            
            first_window = list(tokenized_results.keys())[0]
            tokenizer = tokenized_results[first_window]['tokenizer']
            
            print("\nSUMMARY:")
            print(f"  Processed {len(tokenized_results)} time windows")
            total_sequences = sum(r['stats']['total_sequences'] for r in tokenized_results.values())
            print(f"  Total sequences tokenized: {total_sequences:,}")
            print(f"  Output directory: {OUTPUT_DIR}")
            
            # Show first sequence 
            if tokenized_results:
                demo_window = list(tokenized_results.keys())[0]
                demo_tokenizer = tokenized_results[demo_window]['tokenizer']
                demo_data = tokenized_results[demo_window]['tokenized_data']
                
                print(f"\ tokens from {demo_window} window (first 20 non-PAD tokens):")
                first_seq = demo_data['input_ids'][0]
                decoded_tokens = demo_tokenizer.decode(first_seq)
                non_pad_count = 0
                for i, token in enumerate(decoded_tokens):
                    if token != 'PAD' and non_pad_count < 20:
                        print(f"    [{i:3d}] {token}")
                        non_pad_count += 1
        
        print(f"\nEnd time: {datetime.now()}")
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Please run the preprocessing script first to generate the data.") # User please preprocess the CAIDA preprocessing file before attempting tokenization
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\nTokenization finished.")


if __name__ == "__main__":
    main()