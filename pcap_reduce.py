import os
import sys
from pathlib import Path

def reduce_pcap_with_dpkt(input_file: str, output_file: str, target_size_mb: float = 25):
    """Reduce PCAP file using dpkt (much faster than scapy) to target size 25MB """
    try:
        import dpkt
    except ImportError:
        print(" dpkt not installed. Please run: pip install dpkt")
        sys.exit(1)
    
    print(f" Input file: {input_file}")
    print(f" Target size: {target_size_mb} MB")
    print("-" * 50)
    
    if not os.path.exists(input_file):
        print(f" Input file not found: {input_file}")
        print(f"   Current directory: {os.getcwd()}")
        sys.exit(1)
    
    input_size_bytes = os.path.getsize(input_file)
    input_size_mb = input_size_bytes / (1024 * 1024)
    print(f"Input file size: {input_size_mb:.2f} MB ({input_size_bytes:,} bytes)")
    
    if input_size_mb <= target_size_mb:
        print(f" Input file is already {input_size_mb:.2f} MB (smaller than target)")
        return
    
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f" Created output directory: {output_dir}")
    
    target_size_bytes = target_size_mb * 1024 * 1024
    
    print(f"\n Processing PCAP file with dpkt...")
    print("-" * 50)
    
    try:
        with open(input_file, 'rb') as f_in:
            pcap_reader = dpkt.pcap.Reader(f_in)
            
            with open(output_file, 'wb') as f_out:
                pcap_writer = dpkt.pcap.Writer(f_out)
                
                current_size = 24  # PCAP global header size
                packet_count = 0
                skipped_count = 0
                
                for timestamp, packet_data in pcap_reader:
                    # Calculate size packet data + pcap packet header (16 bytes)
                    packet_size = len(packet_data) + 16
                    
                    # Check if adding this packet would exceed target
                    if current_size + packet_size > target_size_bytes:
                        skipped_count += 1
                        continue  # Count remaining packets
                    
                    # Write packet to output file
                    pcap_writer.writepkt(packet_data, timestamp)
                    current_size += packet_size
                    packet_count += 1
                    
                    # Progress indicator every 10,000 packets
                    if packet_count % 10000 == 0:
                        size_mb = current_size / (1024 * 1024)
                        percent = (current_size / target_size_bytes) * 100
                        print(f"   Processed {packet_count:,} packets | {size_mb:.2f} MB | {percent:.1f}%")
                
        # Final statistics
        print("-" * 50)
        actual_size = os.path.getsize(output_file)
        actual_size_mb = actual_size / (1024 * 1024)
        
        print(f"\n SUCCESS! Reduced PCAP file created")
        print(f"Statistics:")
        print(f"   Output file: {output_file}")
        print(f"   Final size: {actual_size_mb:.2f} MB ({actual_size:,} bytes)")
        print(f"   Packets included: {packet_count:,}")
        print(f"   Reduction ratio: {(actual_size_mb/input_size_mb)*100:.1f}%")
        print(f"   Size reduction: {input_size_mb - actual_size_mb:.2f} MB saved")
        
        return True
        
    except Exception as e:
        print(f"\n Error during processing: {e}")
        # Clean up partial file if it exists
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
                print(f" Cleaned up partial output file")
            except:
                pass
        return False

def main():
    input_file = r"code/data/raw/equinix-nyc.dirA.20190117-130500.UTC.anon.pcap"
    
    output_dir = r"api/test"
    output_filename = "equinix-nyc.dirA.20190117-130500.UTC.anon_25MB.pcap"
    output_file = os.path.join(output_dir, output_filename)
    
    input_file = os.path.normpath(input_file)
    output_file = os.path.normpath(output_file)
    
    # Run the reduction
    success = reduce_pcap_with_dpkt(input_file, output_file, target_size_mb=25)
    
    if success:
        print("\n" + "=" * 50)
        print(" File ready for upload to the deployed API")
        print("=" * 50)
    else:
        print("\n Reduction failed ")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)