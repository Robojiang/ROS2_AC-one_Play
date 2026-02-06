import pyrealsense2 as rs

def get_realsense_devices():
    # 创建上下文对象，它管理所有连接的设备
    ctx = rs.context()
    
    #查询连接的设备列表
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("未检测到 RealSense 设备。请检查 USB 连接。")
        return

    print(f"共检测到 {len(devices)} 个 RealSense 设备:")
    print("="*50)
    
    for i, dev in enumerate(devices):
        try:
            # 获取设备信息
            name = dev.get_info(rs.camera_info.name)
            serial = dev.get_info(rs.camera_info.serial_number)
            fw = dev.get_info(rs.camera_info.firmware_version)
            usb_type = dev.get_info(rs.camera_info.usb_type_descriptor)
            product_id = dev.get_info(rs.camera_info.product_id)
            
            print(f"设备索引: {i}")
            print(f"  设备名称: {name}")
            print(f"  序列号 (SN): {serial}")
            print(f"  固件版本: {fw}")
            print(f"  USB 类型: {usb_type}")
            print(f"  产品 ID: {product_id}")
            print("="*50)
            
        except RuntimeError as e:
            print(f"读取设备 {i} 信息时出错: {e}")
            print("="*50)

if __name__ == "__main__":
    get_realsense_devices()
