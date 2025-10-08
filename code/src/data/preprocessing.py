# Network Traffic Preprocessing for Limited CAIDA Data 
# Handles 1-hour data span with gaps, maximizes sequence generation

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Tuple, Dict, Any, List, Optional
import pickle
from pathlib import Path
import logging
from datetime import datetime, timedelta
import warnings
import ipaddress
from collections import defaultdict
import hashlib
import gc
import os
from scapy.utils import RawPcapReader
from scapy.all import IP, TCP, UDP, ICMP
from scapy.layers.inet6 import IPv6
import psutil

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NetworkTrafficPreprocessor:
    """
    Preprocessor for 1-hour CAIDA data with gaps
    Maximizes sequence generation through:
    1. Very short time windows 
    2. Aggressive overlapping
    3. Direction-aware processing (dirA/dirB)
    4. Gap handling
    5. Multiple augmentation strategies
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.is_fitted = False
        self.ip_anonymization_map = {}
        
        # Memory monitoring
        self.process = psutil.Process()
        self.max_memory_mb = config.get('max_memory_mb', 8192)
        
        # Statistics tracking
        self.global_stats = {
            'total_packets': 0,
            'total_flows': 0,
            'total_bytes': 0,
            'files_processed': 0,
            'dirA_files': 0,
            'dirB_files': 0,
            'time_gaps': []
        }
        
    def check_memory(self) -> float:
        return self.process.memory_info().rss / 1024 / 1024
    
    def process_pcap_fast(self, pcap_path: str, output_path: Path) -> Dict[str, Any]:
        """
        Fast PCAP processing optimized for 1 hour data
        Now with proper error handling for corrupted files
        """
        pcap_name = Path(pcap_path).name
        is_dirA = 'dirA' in pcap_name
        is_dirB = 'dirB' in pcap_name
        
        logger.info(f"Processing {pcap_name} (Direction: {'A' if is_dirA else 'B'})")
        
        try:
            reader = RawPcapReader(pcap_path)
        except Exception as e:
            logger.error(f"Failed to open PCAP file {pcap_name}: {str(e)}")
            return {'flows_file': None, 'stats': {}}
        
        flows = {}
        total_packets = 0
        min_ts = float('inf')
        max_ts = float('-inf')
        
        max_packets = self.config.get('max_packets_per_file', 5000000)
        
        try:
            for i, (pkt_bytes, _) in enumerate(reader):
                if i >= max_packets:
                    break
                
                try:
                    # Parse packet
                    try:
                        pkt = IP(pkt_bytes)
                    except:
                        try:
                            pkt = IPv6(pkt_bytes)
                        except:
                            continue
                    
                    # Extract core features
                    src_ip = self._anonymize_ip(str(pkt.src))
                    dst_ip = self._anonymize_ip(str(pkt.dst))
                    protocol = pkt.proto if hasattr(pkt, 'proto') else pkt.nh
                    length = len(pkt)
                    timestamp = float(pkt.time) if hasattr(pkt, 'time') else 0
                    
                    min_ts = min(min_ts, timestamp)
                    max_ts = max(max_ts, timestamp)
                    
                    # Extract ports
                    src_port = dst_port = 0
                    if pkt.haslayer(TCP):
                        tcp = pkt[TCP]
                        src_port = tcp.sport
                        dst_port = tcp.dport
                        protocol_name = 'TCP'
                    elif pkt.haslayer(UDP):
                        udp = pkt[UDP]
                        src_port = udp.sport
                        dst_port = udp.dport
                        protocol_name = 'UDP'
                    else:
                        protocol_name = f'PROTO_{protocol}'
                    
                    # Create flow key with direction awareness
                    flow_key = f"{src_ip}_{dst_ip}_{src_port}_{dst_port}_{protocol}"
                    
                    if flow_key not in flows:
                        flows[flow_key] = {
                            'src_ip': src_ip,
                            'dst_ip': dst_ip,
                            'src_port': src_port,
                            'dst_port': dst_port,
                            'protocol': protocol,
                            'protocol_name': protocol_name,
                            'direction': 'A' if is_dirA else 'B',
                            'start_time': timestamp,
                            'end_time': timestamp,
                            'packet_count': 0,
                            'total_bytes': 0,
                            'packet_sizes': [],
                            'timestamps': []
                        }
                    
                    flow = flows[flow_key]
                    flow['end_time'] = max(flow['end_time'], timestamp)
                    flow['start_time'] = min(flow['start_time'], timestamp)
                    flow['packet_count'] += 1
                    flow['total_bytes'] += length
                    
                    if len(flow['packet_sizes']) < 50:  # Reduced for speed
                        flow['packet_sizes'].append(length)
                    if len(flow['timestamps']) < 50:
                        flow['timestamps'].append(timestamp)
                    
                    total_packets += 1
                    
                except Exception:
                    continue
                    
        except Exception as e:
            logger.error(f"Error reading packets from {pcap_name}: {str(e)}")
            # If we have some flows, continue with them, otherwise return empty
            if not flows:
                return {'flows_file': None, 'stats': {}}
        
        # Convert flows to DataFrame
        flow_records = []
        for flow_key, flow in flows.items():
            duration = max(flow['end_time'] - flow['start_time'], 0.001)
            packet_sizes = flow['packet_sizes']
            
            record = {
                'flow_id': flow_key,
                'src_ip': flow['src_ip'],
                'dst_ip': flow['dst_ip'],
                'src_port': flow['src_port'],
                'dst_port': flow['dst_port'],
                'protocol': flow['protocol'],
                'protocol_name': flow['protocol_name'],
                'direction': flow['direction'],
                'start_time': flow['start_time'],
                'end_time': flow['end_time'],
                'duration': duration,
                'packet_count': flow['packet_count'],
                'total_bytes': flow['total_bytes'],
                'mean_packet_size': np.mean(packet_sizes) if packet_sizes else 0,
                'std_packet_size': np.std(packet_sizes) if len(packet_sizes) > 1 else 0,
                'packet_rate': flow['packet_count'] / duration,
                'byte_rate': flow['total_bytes'] / duration,
                'source_file': pcap_name
            }
            flow_records.append(record)
        
        if flow_records:
            df = pd.DataFrame(flow_records)
            output_file = output_path / f"flows_{Path(pcap_path).stem}.parquet"
            
            try:
                df.to_parquet(output_file, compression='snappy')
            except Exception as e:
                logger.error(f"Failed to save flows to {output_file}: {str(e)}")
                return {'flows_file': None, 'stats': {}}
            
            time_span = max_ts - min_ts if max_ts > min_ts else 0
            logger.info(f"  Extracted {len(df)} flows, {total_packets} packets, "
                    f"time span: {time_span:.1f}s")
            
            return {
                'flows_file': output_file,
                'stats': {
                    'flows': len(df),
                    'packets': total_packets,
                    'bytes': df['total_bytes'].sum(),
                    'time_span': time_span,
                    'min_timestamp': min_ts,
                    'max_timestamp': max_ts,
                    'direction': 'A' if is_dirA else 'B'
                }
            }
        
        # No flows extracted
        logger.warning(f"No flows extracted from {pcap_name}")
        return {'flows_file': None, 'stats': {}}

    def _anonymize_ip(self, ip_str: str) -> str:
        """Fast IP anonymization"""
        if ip_str not in self.ip_anonymization_map:
            if len(self.ip_anonymization_map) > 500000:
                self.ip_anonymization_map = dict(list(self.ip_anonymization_map.items())[-250000:])
            
            hash_hex = hashlib.md5(ip_str.encode()).hexdigest()
            self.ip_anonymization_map[ip_str] = f"10.{int(hash_hex[:2], 16)}.{int(hash_hex[2:4], 16)}.{int(hash_hex[4:6], 16)}"
        
        return self.ip_anonymization_map[ip_str]
    
    def temporal_aggregation(self, flow_files: List[Path], output_dir: Path) -> Dict[str, Path]:
        """
        temporal aggregation for 1-hour data
        Uses very short windows to maximize samples
        """
        # Use very short windows for 1-hour data
        windows = self.config.get('time_windows', ['30s', '1min','5min'])
        
        logger.info(f"Temporal aggregation for windows: {windows}")
        
        # Load all flows
        all_flows = []
        for flow_file in flow_files:
            df = pd.read_parquet(flow_file)
            all_flows.append(df)
        
        combined = pd.concat(all_flows, ignore_index=True)
        logger.info(f"  Combined {len(combined)} flows from {len(flow_files)} files")
        
        # Analyze time coverage
        time_coverage = combined.groupby('source_file')['start_time'].agg(['min', 'max'])
        logger.info(f"  Time coverage by file:\n{time_coverage}")
        
        # Convert timestamps
        combined['datetime'] = pd.to_datetime(combined['start_time'], unit='s')
        
        aggregated_files = {}
        
        for window in windows:
            logger.info(f"\n  Aggregating {window} window")
            window_dir = output_dir / f"{window}_window"
            window_dir.mkdir(exist_ok=True)
            
            df_indexed = combined.set_index('datetime')
            
            # Check which columns are actually available
            available_columns = df_indexed.columns.tolist()
            logger.info(f"    Available columns: {available_columns[:10]}...")  # Show first 10 columns
            
            # Simplified aggregation for speed - only use available columns
            agg_functions = {}
            
            if 'packet_count' in available_columns:
                agg_functions['packet_count'] = ['sum', 'mean', 'max']
            if 'total_bytes' in available_columns:
                agg_functions['total_bytes'] = ['sum', 'mean']
            if 'mean_packet_size' in available_columns:
                agg_functions['mean_packet_size'] = 'mean'
            if 'packet_rate' in available_columns:
                agg_functions['packet_rate'] = 'mean'
            if 'byte_rate' in available_columns:
                agg_functions['byte_rate'] = 'mean'
            if 'flow_id' in available_columns:
                agg_functions['flow_id'] = 'nunique'
            if 'src_ip' in available_columns:
                agg_functions['src_ip'] = 'nunique'
            if 'dst_ip' in available_columns:
                agg_functions['dst_ip'] = 'nunique'
            if 'direction' in available_columns:
                agg_functions['direction'] = lambda x: 'BOTH' if len(x) > 0 and x.nunique() > 1 else (x.iloc[0] if len(x) > 0 else 'UNKNOWN')
            
            if not agg_functions:
                logger.error(f"    No suitable columns found for aggregation in {available_columns}")
                continue
            
            logger.info(f"    Using aggregation functions for: {list(agg_functions.keys())}")
            
            # Perform aggregation
            try:
                aggregated = df_indexed.resample(window).agg(agg_functions)
                
                # Flatten columns - handle both tuple and string column names
                new_columns = []
                for col in aggregated.columns:
                    if isinstance(col, tuple):
                        # Multi-level column from aggregation
                        new_col_name = '_'.join(str(c) for c in col if str(c) != '')
                    else:
                        # Single level column
                        new_col_name = str(col)
                    new_columns.append(new_col_name)
                
                aggregated.columns = new_columns
                logger.info(f"    Aggregated columns: {aggregated.columns.tolist()[:10]}...")
                
                # Find the packet count column (could be 'packet_count_sum' or similar)
                packet_count_cols = [col for col in aggregated.columns if 'packet_count' in col and 'sum' in col]
                if not packet_count_cols:
                    # Try alternative patterns
                    packet_count_cols = [col for col in aggregated.columns if 'packet' in col and ('sum' in col or 'count' in col)]
                
                if packet_count_cols:
                    primary_filter_col = packet_count_cols[0]
                    logger.info(f"    Using '{primary_filter_col}' to filter empty windows")
                    # Remove empty windows
                    aggregated = aggregated[aggregated[primary_filter_col] > 0]
                else:
                    logger.warning(f"    No packet count column found for filtering. Available columns: {aggregated.columns.tolist()}")
                    # Use any numeric column with positive values for filtering
                    numeric_cols = aggregated.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        first_numeric = numeric_cols[0]
                        logger.info(f"    Using '{first_numeric}' as fallback filter")
                        aggregated = aggregated[aggregated[first_numeric] > 0]
                
                # Fill gaps with interpolation
                aggregated = aggregated.fillna(method='ffill').fillna(method='bfill').fillna(0)
                
                # Add minimal features
                aggregated = self._add_minimal_features(aggregated)
                
                # Save
                aggregated = aggregated.reset_index()
                aggregated['time_window'] = window
                
                final_file = window_dir / "aggregated_data.parquet"
                aggregated.to_parquet(final_file, compression='snappy')
                aggregated.to_csv(window_dir / "aggregated_data.csv", index=False)
                
                logger.info(f"    Created {len(aggregated)} samples for {window} window")
                aggregated_files[window] = final_file
                
            except Exception as e:
                logger.error(f"    Error during aggregation for {window}: {str(e)}")
                import traceback
                logger.debug(f"    Full traceback: {traceback.format_exc()}")
                continue
            
            del df_indexed, aggregated
            gc.collect()
    
        return aggregated_files
    
    def _add_minimal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add minimal but essential features"""
        # Time features
        df['timestamp_numeric'] = df.index.astype(np.int64) // 10**9
        df['time_since_start'] = df['timestamp_numeric'] - df['timestamp_numeric'].min()
        
        # Simple cyclical encoding
        seconds_in_hour = (df['time_since_start'] % 3600)
        df['hour_sin'] = np.sin(2 * np.pi * seconds_in_hour / 3600)
        df['hour_cos'] = np.cos(2 * np.pi * seconds_in_hour / 3600)
        
        # Ratios
        if 'packet_count_sum' in df.columns and 'flow_id_nunique' in df.columns:
            df['packets_per_flow'] = df['packet_count_sum'] / (df['flow_id_nunique'] + 1)
        
        # Simple moving features (very small windows)
        key_metrics = ['packet_count_sum', 'total_bytes_sum']
        for metric in key_metrics:
            if metric in df.columns:
                df[f'{metric}_ma2'] = df[metric].rolling(window=2, min_periods=1).mean()
                df[f'{metric}_diff'] = df[metric].diff()
                df[f'{metric}_lag1'] = df[metric].shift(1)
        
        return df.fillna(0)
    
    def create_maximum_sequences(self, aggregated_file: Path, output_dir: Path) -> Dict[str, Any]:
        """
        Create maximum sequences using aggressive strategies
        handles different sequence lengths properly
        """
        logger.info(f"Creating maximum sequences from {aggregated_file}")
        
        df = pd.read_parquet(aggregated_file)
        n_samples = len(df)
        logger.info(f"  Available samples: {n_samples}")
        
        if n_samples < 10:
            logger.warning(f"  Too few samples ({n_samples} < 10)")
            return self._create_empty_result(n_samples)
        
        # Sort by time
        if 'datetime' in df.columns:
            df = df.sort_values('datetime')
        
        # Aggressive sequence parameters for limited data
        configs = []
        
        if n_samples >= 50:
            configs.append({'seq_len': 8, 'horizon': 2, 'stride': 1})
            configs.append({'seq_len': 6, 'horizon': 2, 'stride': 1})
            configs.append({'seq_len': 4, 'horizon': 1, 'stride': 1})
        elif n_samples >= 30:
            configs.append({'seq_len': 5, 'horizon': 1, 'stride': 1})
            configs.append({'seq_len': 4, 'horizon': 1, 'stride': 1})
            configs.append({'seq_len': 3, 'horizon': 1, 'stride': 1})
        elif n_samples >= 20:
            configs.append({'seq_len': 3, 'horizon': 1, 'stride': 1})
            configs.append({'seq_len': 2, 'horizon': 1, 'stride': 1})
        else:
            configs.append({'seq_len': 2, 'horizon': 1, 'stride': 1})
        
        # Normalize
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = ['datetime', 'timestamp_numeric']
        numeric_cols = [col for col in numeric_cols if col not in exclude]
        
        scaler = RobustScaler()
        df_normalized = df.copy()
        
        if numeric_cols:
            df_normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        # Save normalized data
        df_normalized.to_csv(output_dir / "normalized_data.csv", index=False)
        
        # Target columns
        target_cols = ['packet_count_sum', 'total_bytes_sum']
        target_cols = [col for col in target_cols if col in df.columns]
        if not target_cols:
            target_cols = numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols
        
        features = df_normalized[numeric_cols].values if numeric_cols else df_normalized.values
        targets = df_normalized[target_cols].values if target_cols else features[:, :2]
        
        # Process each configuration separately and save individually
        all_results = {}
        best_config = None
        max_sequences = 0
        
        for i, config in enumerate(configs):
            seq_len = config['seq_len']
            horizon = config['horizon']
            stride = config['stride']
            
            if n_samples >= seq_len + horizon:
                sequences = []
                labels = []
                
                for j in range(0, n_samples - seq_len - horizon + 1, stride):
                    sequences.append(features[j:j + seq_len])
                    labels.append(targets[j + seq_len:j + seq_len + horizon])
                
                if sequences:
                    sequences = np.array(sequences)
                    labels = np.array(labels)
                    
                    # Add slight variation to create diversity
                    if i > 0:
                        noise = np.random.normal(0, 0.005 * i, sequences.shape)
                        sequences = sequences + noise
                    
                    # Store this configuration's results
                    config_key = f"config_{i+1}_seq{seq_len}_h{horizon}"
                    all_results[config_key] = {
                        'sequences': sequences,
                        'labels': labels,
                        'config': config,
                        'n_sequences': len(sequences)
                    }
                    
                    logger.info(f"    Config {i+1}: seq_len={seq_len}, horizon={horizon} "
                            f"→ {len(sequences)} sequences")
                    
                    # Track the best configuration
                    if len(sequences) > max_sequences:
                        max_sequences = len(sequences)
                        best_config = config_key
        
        #save each configuration separately
        total_sequences_created = 0
        
        for config_key, result in all_results.items():
            sequences = result['sequences']
            labels = result['labels']
            config = result['config']
            
            # Save individual configuration
            np.savez_compressed(
                output_dir / f"lstm_data_{config_key}.npz",
                sequences=sequences,
                labels=labels,
                feature_names=numeric_cols,
                target_names=target_cols,
                config=config
            )
            
            total_sequences_created += len(sequences)
            
            # Additional augmentation for the best configuration if still too few
            if config_key == best_config and len(sequences) < 100:
                aug_sequences = []
                aug_labels = []
                
                # Time-shift augmentation
                for shift in [0.01, -0.01, 0.02]:
                    shifted = sequences + shift
                    aug_sequences.append(shifted)
                    aug_labels.append(labels)
                
                # Scaling augmentation
                for scale in [0.98, 1.02]:
                    scaled = sequences * scale
                    aug_sequences.append(scaled)
                    aug_labels.append(labels)
                
                augmented_sequences = np.vstack([sequences] + aug_sequences)
                augmented_labels = np.vstack([labels] + aug_labels)
                
                # Save augmented version
                np.savez_compressed(
                    output_dir / f"lstm_data_{config_key}_augmented.npz",
                    sequences=augmented_sequences,
                    labels=augmented_labels,
                    feature_names=numeric_cols,
                    target_names=target_cols,
                    config=config
                )
                
                logger.info(f"    Augmented {config_key} to {len(augmented_sequences)} sequences")
                total_sequences_created += len(aug_sequences) * len(aug_labels)
        
        # Create a combined dataset from the best configuration for convenience
        if best_config and best_config in all_results:
            best_result = all_results[best_config]
            
            # Save the best configuration as the primary dataset
            np.savez_compressed(
                output_dir / "lstm_data.npz",
                sequences=best_result['sequences'],
                labels=best_result['labels'],
                feature_names=numeric_cols,
                target_names=target_cols,
                config=best_result['config']
            )
        
        # Create ARIMA data
        if 'datetime' in df.columns and target_cols:
            arima_df = df[['datetime'] + target_cols].copy()
            arima_df.set_index('datetime', inplace=True)
            arima_df.to_csv(output_dir / "arima_data.csv")
        
        # Save configuration summary
        config_summary = {
            'all_configs': {k: v['config'] for k, v in all_results.items()},
            'sequence_counts': {k: v['n_sequences'] for k, v in all_results.items()},
            'best_config': best_config,
            'total_sequences': total_sequences_created
        }
        
        import json
        with open(output_dir / "sequence_configs.json", 'w') as f:
            json.dump(config_summary, f, indent=2, default=str)
        
        logger.info(f"  FINAL: Created {total_sequences_created} total sequences across {len(all_results)} configurations")
        logger.info(f"  BEST CONFIG: {best_config} with {max_sequences} sequences")
        
        return {
            'n_samples': n_samples,
            'n_sequences': total_sequences_created,
            'n_configs': len(all_results),
            'best_config': best_config,
            'max_sequences_per_config': max_sequences,
            'sequence_shape': all_results[best_config]['sequences'].shape if best_config else (0,),
            'label_shape': all_results[best_config]['labels'].shape if best_config else (0,),
            'configs': [v['config'] for v in all_results.values()],
            'feature_names': numeric_cols
        }
        
    def _create_empty_result(self, n_samples: int) -> Dict[str, Any]:
        """Create empty result for insufficient data"""
        return {
            'n_samples': n_samples,
            'n_sequences': 0,
            'sequence_shape': (0,),
            'label_shape': (0,),
            'configs': [],
            'feature_names': []
        }
    def process_all(self, pcap_directory: str, output_base: str) -> Dict[str, Any]:
        """
        Optimized processing pipeline for 1-hour CAIDA data
        """
        logger.info("="*60)
        logger.info("OPTIMIZED PROCESSING FOR CAIDA DATA")
        logger.info("="*60)
        
        # Initialize directories FIRST
        output_dir = Path(output_base)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        flows_dir = output_dir / "flows"
        flows_dir.mkdir(exist_ok=True)
        
        # NOW check for existing flow files after directories are created
        existing_flows = list(flows_dir.glob("flows_*.parquet"))
        
        # Initialize stats
        file_stats = {}
        
        # Get list of PCAP files for reference
        pcap_files = sorted(Path(pcap_directory).glob("*.pcap"))
        logger.info(f"Found {len(pcap_files)} PCAP files in source directory")
        
        if existing_flows:
            logger.info(f"Found {len(existing_flows)} existing flow files - skipping PCAP processing")
            flow_files = existing_flows
            
            # Log which flow files were found
            for flow_file in existing_flows[:5]:  # Show first 5 as examples
                logger.info(f"  Using existing: {flow_file.name}")
            if len(existing_flows) > 5:
                logger.info(f"  ... and {len(existing_flows) - 5} more files")
                
        else:
            logger.info("No existing flow files found - processing PCAP files")
            
            if not pcap_files:
                raise ValueError(f"No PCAP files found in {pcap_directory}")
            
            # Identify dirA and dirB files
            dirA_files = [f for f in pcap_files if 'dirA' in f.name]
            dirB_files = [f for f in pcap_files if 'dirB' in f.name]
            logger.info(f"  Direction A: {len(dirA_files)} files")
            logger.info(f"  Direction B: {len(dirB_files)} files")
            
            # Process each PCAP file
            flow_files = []
            processed_files = set()  # Track processed files to avoid duplicates
            
            for i, pcap_file in enumerate(pcap_files):
                # Check if file was already processed (safety check)
                if str(pcap_file) in processed_files:
                    logger.warning(f"Skipping already processed file: {pcap_file.name}")
                    continue
                    
                logger.info(f"Processing file {i+1}/{len(pcap_files)}: {pcap_file.name}")
                
                try:
                    # Add to processed set immediately
                    processed_files.add(str(pcap_file))
                    
                    result = self.process_pcap_fast(str(pcap_file), flows_dir)
                    
                    if result['flows_file']:
                        flow_files.append(result['flows_file'])
                        file_stats[pcap_file.name] = result['stats']
                        
                        self.global_stats['total_packets'] += result['stats']['packets']
                        self.global_stats['total_flows'] += result['stats']['flows']
                        self.global_stats['files_processed'] += 1
                        
                        if result['stats']['direction'] == 'A':
                            self.global_stats['dirA_files'] += 1
                        else:
                            self.global_stats['dirB_files'] += 1
                        
                        logger.info(f"   Successfully processed {pcap_file.name}")
                        
                    else:
                        logger.warning(f"  ⚠ No flows extracted from {pcap_file.name}")
                        
                except Exception as e:
                    logger.error(f"  ✗ Error processing {pcap_file.name}: {str(e)}")
                    # Log the full traceback for debugging
                    import traceback
                    logger.debug(f"Full traceback: {traceback.format_exc()}")
                    continue
                
                # Memory cleanup after each file
                gc.collect()
                
                # Log progress every 5 files
                if (i + 1) % 5 == 0:
                    current_memory = self.check_memory()
                    logger.info(f"  Progress: {i+1}/{len(pcap_files)} files, Memory: {current_memory:.0f}MB")
            
            if not flow_files:
                raise ValueError("No flows extracted from any PCAP file")
            
            logger.info(f"Successfully processed {len(flow_files)} files out of {len(pcap_files)} total")
        
        # Now continue with aggregation (same for both paths)
        logger.info("="*60)
        logger.info("PERFORMING TEMPORAL AGGREGATION")
        logger.info("="*60)
        
        # Perform temporal aggregation
        aggregated_files = self.temporal_aggregation(flow_files, output_dir)
        
        if not aggregated_files:
            logger.error("No aggregated files were created")
            return {
                'output_dir': output_dir,
                'processed_windows': {},
                'global_stats': self.global_stats
            }
        
        logger.info("="*60)
        logger.info("CREATING MAXIMUM SEQUENCES")
        logger.info("="*60)
        
        # Process each time window
        processed_windows = {}
        
        for window, agg_file in aggregated_files.items():
            logger.info(f"Processing {window} window")
            
            window_dir = output_dir / f"{window}_window"
            window_dir.mkdir(exist_ok=True)
            
            result = self.create_maximum_sequences(agg_file, window_dir)
            processed_windows[window] = result
        
        # Save metadata
        metadata = {
            'config': self.config,
            'global_stats': self.global_stats,
            'file_stats': file_stats,
            'processed_windows': processed_windows,
            'timestamp': datetime.now().isoformat(),
            'total_files_found': len(pcap_files),
            'flow_files_used': len(flow_files),
            'existing_flows_reused': len(existing_flows) > 0
        }
        
        import json
        with open(output_dir / "preprocessing_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Metadata saved to {output_dir / 'preprocessing_metadata.json'}")
        
        return {
            'output_dir': output_dir,
            'processed_windows': processed_windows,
            'global_stats': self.global_stats
        }


def main():
    # Optimized configuration
    config = {
        # Processing limits
        'max_packets_per_file': 5000000,
        'max_memory_mb': 8192,
        
        # Very short windows for 1-hour data
        'time_windows': ['10s','30s', '1min'],
        
        # Normalization
        'normalization_method': 'robust'
    }
    
    # Initialize preprocessor
    preprocessor = NetworkTrafficPreprocessor(config)
    
    # Paths
    pcap_directory = "code/data/raw"
    output_dir = "code/data/processed_data"
    
    try:
        start_time = datetime.now()
        logger.info(f"Starting at: {start_time}")
        
        # Process all PCAPs
        results = preprocessor.process_all(pcap_directory, output_dir)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        
        print(f"\nPROCESSING TIME: {duration}")
        
        print(f"\nGLOBAL STATISTICS:")
        print(f"  Files processed: {results['global_stats']['files_processed']}")
        print(f"  Direction A files: {results['global_stats']['dirA_files']}")
        print(f"  Direction B files: {results['global_stats']['dirB_files']}")
        print(f"  Total packets: {results['global_stats']['total_packets']:,}")
        print(f"  Total flows: {results['global_stats']['total_flows']:,}")
        
        print(f"\nSEQUENCE GENERATION RESULTS:")
        
        total_sequences = 0
        best_window = None
        best_sequences = 0
        
        for window, data in results['processed_windows'].items():
            print(f"\n  {window.upper()} window:")
            print(f"    Samples available: {data['n_samples']}")
            print(f"    Sequences created: {data['n_sequences']}")
            
            if data['n_sequences'] > 0:
                print(f"    Shape: {data['sequence_shape']}")
                print(f"    Configurations used: {len(data.get('configs', []))}")
                total_sequences += data['n_sequences']
                
                if data['n_sequences'] > best_sequences:
                    best_sequences = data['n_sequences']
                    best_window = window
                
                # Status indicator
                if data['n_sequences'] >= 500:
                    print(f" EXCELLENT for training")
                elif data['n_sequences'] >= 200:
                    print(f" GOOD for training")
                elif data['n_sequences'] >= 100:
                    print(f"  MINIMUM for training")
                else:
                    print(f"  LIMITED - Use with caution")
        
        print(f"\n{'='*60}")
        print(f"TOTAL SEQUENCES GENERATED: {total_sequences:,}")
        print(f"BEST WINDOW: {best_window} with {best_sequences:,} sequences")
        print(f"{'='*60}")
        
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        gc.collect()
        final_memory = preprocessor.check_memory()
        print(f"\nFinal memory usage: {final_memory:.0f}MB")
        print("\nProcessing completed.")


if __name__ == "__main__":
    main()