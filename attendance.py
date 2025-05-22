import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import sys
import cv2
import boto3
import uuid
import json
from decimal import Decimal  # For DynamoDB float conversion
import base64
from datetime import datetime
from PIL import Image, ImageTk
import io
import threading
import time

# AWS Configuration
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
COLLECTION_ID = 'FaceAttendanceCollection'
USERS_TABLE = 'FaceAttendanceUsers'
ATTENDANCE_TABLE = 'FaceAttendanceRecords'

# Initialize AWS clients
rekognition = boto3.client('rekognition', region_name=AWS_REGION)
dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)

# Helper function to convert floats to Decimal for DynamoDB
def float_to_decimal(obj):
    if isinstance(obj, float):
        return Decimal(str(obj))
    elif isinstance(obj, dict):
        return {k: float_to_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [float_to_decimal(i) for i in obj]
    else:
        return obj

# Ensure tables and collection exist
def initialize_aws_resources():
    # Create DynamoDB tables if they don't exist
    tables = list(dynamodb.tables.all())
    table_names = [table.name for table in tables]

    if USERS_TABLE not in table_names:
        users_table = dynamodb.create_table(
            TableName=USERS_TABLE,
            KeySchema=[{'AttributeName': 'user_id', 'KeyType': 'HASH'}],
            AttributeDefinitions=[{'AttributeName': 'user_id', 'AttributeType': 'S'}],
            ProvisionedThroughput={'ReadCapacityUnits': 5, 'WriteCapacityUnits': 5}
        )
        users_table.meta.client.get_waiter('table_exists').wait(TableName=USERS_TABLE)
        print(f"Created {USERS_TABLE} table")

    if ATTENDANCE_TABLE not in table_names:
        attendance_table = dynamodb.create_table(
            TableName=ATTENDANCE_TABLE,
            KeySchema=[
                {'AttributeName': 'user_id', 'KeyType': 'HASH'},
                {'AttributeName': 'timestamp', 'KeyType': 'RANGE'}
            ],
            AttributeDefinitions=[
                {'AttributeName': 'user_id', 'AttributeType': 'S'},
                {'AttributeName': 'timestamp', 'AttributeType': 'S'}
            ],
            ProvisionedThroughput={'ReadCapacityUnits': 5, 'WriteCapacityUnits': 5}
        )
        attendance_table.meta.client.get_waiter('table_exists').wait(TableName=ATTENDANCE_TABLE)
        print(f"Created {ATTENDANCE_TABLE} table")

    # Create Rekognition collection if it doesn't exist
    try:
        collections = rekognition.list_collections()
        if COLLECTION_ID not in collections['CollectionIds']:
            rekognition.create_collection(CollectionId=COLLECTION_ID)
            print(f"Created {COLLECTION_ID} collection")
    except Exception as e:
        print(f"Error creating collection: {e}")
        messagebox.showerror("Error", f"Failed to initialize AWS resources: {e}")
        return False

    return True

# Main Application
class FaceAttendanceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Facial Recognition Attendance System")
        self.geometry("1200x700")
        self.minsize(1000, 600)
        
        # Configure style
        self.style = ttk.Style(self)
        self.style.theme_use('clam')
        
        # Configure colors
        self.bg_color = "#f5f5f5"
        self.accent_color = "#4a6fa5"
        self.configure(bg=self.bg_color)
        
        self.style.configure('TFrame', background=self.bg_color)
        self.style.configure('TLabel', background=self.bg_color, font=('Helvetica', 11))
        self.style.configure('TButton', font=('Helvetica', 11))
        self.style.configure('Accent.TButton', background=self.accent_color)
        self.style.configure('TNotebook', background=self.bg_color)
        self.style.configure('TNotebook.Tab', padding=[10, 5], font=('Helvetica', 11))
        
        # Initialize camera variables
        self.camera = None
        self.is_camera_running = False
        self.current_frame = None
        self.camera_thread = None
        
        # Initialize debug mode variable - FIX: Create as instance variable
        self.debug_mode_var = tk.BooleanVar(value=False)
        
        # Create main container
        self.main_container = ttk.Frame(self)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create header
        self.create_header()
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create tabs
        self.dashboard_tab = ttk.Frame(self.notebook)
        self.register_tab = ttk.Frame(self.notebook)
        self.attendance_tab = ttk.Frame(self.notebook)
        self.users_tab = ttk.Frame(self.notebook)
        self.reports_tab = ttk.Frame(self.notebook)
        self.verification_tab = ttk.Frame(self.notebook)  # New verification tab
        
        self.notebook.add(self.dashboard_tab, text="Dashboard")
        self.notebook.add(self.register_tab, text="Register User")
        self.notebook.add(self.attendance_tab, text="Take Attendance")
        self.notebook.add(self.users_tab, text="Manage Users")
        self.notebook.add(self.reports_tab, text="Reports")
        self.notebook.add(self.verification_tab, text="Verification")  # Add verification tab
        
        # Initialize tabs
        self.init_dashboard()
        self.init_register_tab()
        self.init_attendance_tab()
        self.init_users_tab()
        self.init_reports_tab()
        self.init_verification_tab()  # Initialize verification tab
        
        # Initialize AWS resources
        if not initialize_aws_resources():
            messagebox.showerror("Error", "Failed to initialize AWS resources. Please check your credentials.")
        
        # Bind closing event
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_header(self):
        header_frame = ttk.Frame(self.main_container)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = ttk.Label(header_frame, text="Facial Recognition Attendance System", 
                               font=('Helvetica', 18, 'bold'))
        title_label.pack(side=tk.LEFT)
        
        # Add current date and time
        self.time_label = ttk.Label(header_frame, text="", font=('Helvetica', 12))
        self.time_label.pack(side=tk.RIGHT)
        
        # Add debug mode toggle - FIX: Use self.debug_mode_var instead of self.debug_mode
        debug_frame = ttk.Frame(header_frame)
        debug_frame.pack(side=tk.RIGHT, padx=20)
        
        debug_check = ttk.Checkbutton(debug_frame, text="Debug Mode", variable=self.debug_mode_var)
        debug_check.pack(side=tk.RIGHT)
        
        self.update_time()

    def update_time(self):
        current_time = datetime.now().strftime("%B %d, %Y %H:%M:%S")
        self.time_label.config(text=current_time)
        self.after(1000, self.update_time)

    def init_dashboard(self):
        # Create dashboard layout
        dashboard_frame = ttk.Frame(self.dashboard_tab)
        dashboard_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Welcome message
        welcome_frame = ttk.Frame(dashboard_frame)
        welcome_frame.pack(fill=tk.X, pady=20)
        
        welcome_label = ttk.Label(welcome_frame, 
                                 text="Welcome to the Facial Recognition Attendance System",
                                 font=('Helvetica', 16, 'bold'))
        welcome_label.pack()
        
        desc_label = ttk.Label(welcome_frame, 
                              text="Track attendance automatically using facial recognition technology",
                              font=('Helvetica', 12))
        desc_label.pack(pady=5)
        
        # Stats cards
        stats_frame = ttk.Frame(dashboard_frame)
        stats_frame.pack(fill=tk.X, pady=20)
        
        # Create 4 stat cards in a row
        self.create_stat_card(stats_frame, "Total Users", "0", 0)
        self.create_stat_card(stats_frame, "Present Today", "0", 1)
        self.create_stat_card(stats_frame, "Absent Today", "0", 2)
        self.create_stat_card(stats_frame, "Attendance Rate", "0%", 3)
        
        # Quick actions
        actions_frame = ttk.Frame(dashboard_frame)
        actions_frame.pack(fill=tk.X, pady=20)
        
        actions_label = ttk.Label(actions_frame, text="Quick Actions", font=('Helvetica', 14, 'bold'))
        actions_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Create action buttons
        actions_buttons_frame = ttk.Frame(actions_frame)
        actions_buttons_frame.pack(fill=tk.X)
        
        register_btn = ttk.Button(actions_buttons_frame, text="Register New User",
                                 command=lambda: self.notebook.select(self.register_tab))
        register_btn.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        
        attendance_btn = ttk.Button(actions_buttons_frame, text="Take Attendance",
                                   command=lambda: self.notebook.select(self.attendance_tab))
        attendance_btn.grid(row=0, column=1, padx=10, pady=10, sticky=tk.W)
        
        users_btn = ttk.Button(actions_buttons_frame, text="Manage Users",
                              command=lambda: self.notebook.select(self.users_tab))
        users_btn.grid(row=0, column=2, padx=10, pady=10, sticky=tk.W)
        
        reports_btn = ttk.Button(actions_buttons_frame, text="View Reports",
                                command=lambda: self.notebook.select(self.reports_tab))
        reports_btn.grid(row=0, column=3, padx=10, pady=10, sticky=tk.W)
        
        # Add verification button
        verify_btn = ttk.Button(actions_buttons_frame, text="Verify Face",
                               command=lambda: self.notebook.select(self.verification_tab))
        verify_btn.grid(row=0, column=4, padx=10, pady=10, sticky=tk.W)
        
        # System status
        status_frame = ttk.Frame(dashboard_frame)
        status_frame.pack(fill=tk.X, pady=20)
        
        status_label = ttk.Label(status_frame, text="System Status", font=('Helvetica', 14, 'bold'))
        status_label.pack(anchor=tk.W, pady=(0, 10))
        
        # AWS connection status
        aws_frame = ttk.Frame(status_frame)
        aws_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(aws_frame, text="AWS Connection:").pack(side=tk.LEFT)
        self.aws_status = ttk.Label(aws_frame, text="Checking...", foreground="orange")
        self.aws_status.pack(side=tk.LEFT, padx=10)
        
        # Check AWS connection
        self.check_aws_connection()
        
        # Update dashboard stats
        self.update_dashboard_stats()

    def check_aws_connection(self):
        try:
            # Test AWS connection by listing collections
            rekognition.list_collections()
            self.aws_status.config(text="Connected", foreground="green")
        except Exception as e:
            self.aws_status.config(text=f"Error: {str(e)[:50]}...", foreground="red")
            # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
            if self.debug_mode_var.get():
                print(f"AWS Connection Error: {e}")

    def create_stat_card(self, parent, title, value, column):
        card_frame = ttk.Frame(parent, style='Card.TFrame')
        card_frame.grid(row=0, column=column, padx=10, sticky=tk.W+tk.E)
        parent.columnconfigure(column, weight=1)
        
        # Add border and padding
        card_inner = ttk.Frame(card_frame, padding=15)
        card_inner.pack(fill=tk.BOTH, expand=True)
        
        # Add title and value
        title_label = ttk.Label(card_inner, text=title, font=('Helvetica', 12))
        title_label.pack(anchor=tk.W)
        
        value_label = ttk.Label(card_inner, text=value, font=('Helvetica', 24, 'bold'))
        value_label.pack(anchor=tk.W, pady=5)
        
        # Store reference to update later
        card_frame.value_label = value_label
        return card_frame

    def update_dashboard_stats(self):
        try:
            # Get total users
            users_table = dynamodb.Table(USERS_TABLE)
            user_count = users_table.scan(Select='COUNT')['Count']
            
            # Get today's attendance
            attendance_table = dynamodb.Table(ATTENDANCE_TABLE)
            today = datetime.now().strftime("%Y-%m-%d")
            
            # Scan for today's records
            response = attendance_table.scan(
                FilterExpression="begins_with(#ts, :today)",
                ExpressionAttributeNames={"#ts": "timestamp"},
                ExpressionAttributeValues={":today": today}
            )
            
            # Count unique users who attended today
            present_users = set()
            for item in response.get('Items', []):
                present_users.add(item['user_id'])
            
            present_count = len(present_users)
            absent_count = max(0, user_count - present_count)
            
            # Calculate attendance rate
            attendance_rate = 0
            if user_count > 0:
                attendance_rate = (present_count / user_count) * 100
                attendance_rate = Decimal(str(attendance_rate))  # Convert to Decimal
            
            # Update stat cards
            for child in self.dashboard_tab.winfo_children():
                if isinstance(child, ttk.Frame):
                    for grandchild in child.winfo_children():
                        if isinstance(grandchild, ttk.Frame) and hasattr(grandchild, 'winfo_children'):
                            for stat_card in grandchild.winfo_children():
                                if hasattr(stat_card, 'value_label'):
                                    title = stat_card.winfo_children()[0].winfo_children()[0].cget('text')
                                    if title == "Total Users":
                                        stat_card.value_label.config(text=str(user_count))
                                    elif title == "Present Today":
                                        stat_card.value_label.config(text=str(present_count))
                                    elif title == "Absent Today":
                                        stat_card.value_label.config(text=str(absent_count))
                                    elif title == "Attendance Rate":
                                        stat_card.value_label.config(text=f"{float(attendance_rate):.1f}%")
        except Exception as e:
            # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
            if self.debug_mode_var.get():
                print(f"Error updating dashboard stats: {e}")
                import traceback
                traceback.print_exc()
        
        # Schedule next update
        self.after(60000, self.update_dashboard_stats)  # Update every minute

    def init_register_tab(self):
        register_frame = ttk.Frame(self.register_tab)
        register_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(register_frame, text="Register New User", font=('Helvetica', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 20))
        
        # Create two columns
        left_frame = ttk.Frame(register_frame)
        left_frame.grid(row=1, column=0, sticky=tk.NSEW, padx=(0, 10))
        
        right_frame = ttk.Frame(register_frame)
        right_frame.grid(row=1, column=1, sticky=tk.NSEW, padx=(10, 0))
        
        register_frame.columnconfigure(0, weight=1)
        register_frame.columnconfigure(1, weight=1)
        register_frame.rowconfigure(1, weight=1)
        
        # Camera frame (left side)
        camera_label = ttk.Label(left_frame, text="Capture Face", font=('Helvetica', 14, 'bold'))
        camera_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Create a container for camera and buttons with proper layout
        camera_container = ttk.Frame(left_frame)
        camera_container.pack(fill=tk.BOTH, expand=True)
        
        # Camera frame with fixed height to prevent overlapping buttons
        self.register_camera_frame = ttk.Frame(camera_container, borderwidth=1, relief=tk.SOLID)
        self.register_camera_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.register_camera_label = ttk.Label(self.register_camera_frame)
        self.register_camera_label.pack(fill=tk.BOTH, expand=True)
        
        # Separate frame for buttons with fixed position
        camera_buttons_frame = ttk.Frame(camera_container)
        camera_buttons_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.register_start_camera_btn = ttk.Button(camera_buttons_frame, text="Start Camera", 
                                                  command=lambda: self.start_camera(self.register_camera_label))
        self.register_start_camera_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.register_capture_btn = ttk.Button(camera_buttons_frame, text="Capture Image", 
                                             command=self.capture_register_image)
        self.register_capture_btn.pack(side=tk.LEFT, padx=5)
        
        # Add the new "Load Image" button
        self.register_load_btn = ttk.Button(camera_buttons_frame, text="Load Image", 
                                          command=self.load_image_for_registration)
        self.register_load_btn.pack(side=tk.LEFT, padx=5)
        
        self.register_reset_btn = ttk.Button(camera_buttons_frame, text="Reset", 
                                           command=self.reset_register_image, state=tk.DISABLED)
        self.register_reset_btn.pack(side=tk.LEFT, padx=5)
        
        # User info form (right side)
        form_label = ttk.Label(right_frame, text="User Information", font=('Helvetica', 14, 'bold'))
        form_label.pack(anchor=tk.W, pady=(0, 10))
        
        form_frame = ttk.Frame(right_frame, padding=15)
        form_frame.pack(fill=tk.BOTH, expand=True)
        
        # User ID
        ttk.Label(form_frame, text="User ID:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.register_user_id = ttk.Entry(form_frame, width=30)
        self.register_user_id.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # Full Name
        ttk.Label(form_frame, text="Full Name:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.register_name = ttk.Entry(form_frame, width=30)
        self.register_name.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # Email
        ttk.Label(form_frame, text="Email:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.register_email = ttk.Entry(form_frame, width=30)
        self.register_email.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # Department
        ttk.Label(form_frame, text="Department:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.register_department = ttk.Combobox(form_frame, width=28, 
                                              values=["Engineering", "Marketing", "Sales", "HR", "Finance", "IT", "Other"])
        self.register_department.grid(row=3, column=1, sticky=tk.W, pady=5)
        
        # Face quality threshold
        ttk.Label(form_frame, text="Quality Threshold:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.quality_threshold = ttk.Scale(form_frame, from_=70, to=99, orient=tk.HORIZONTAL)
        self.quality_threshold.set(80)  # Default value
        self.quality_threshold.grid(row=4, column=1, sticky=tk.W, pady=5)
        
        # Threshold value display
        self.threshold_value = ttk.Label(form_frame, text="80%")
        self.threshold_value.grid(row=4, column=2, sticky=tk.W, pady=5)
        
        # Update threshold value display when slider moves
        self.quality_threshold.config(command=self.update_threshold_display)
        
        # Status message
        self.register_status = ttk.Label(form_frame, text="", font=('Helvetica', 10))
        self.register_status.grid(row=5, column=0, columnspan=3, sticky=tk.W, pady=10)
        
        # Register button
        self.register_btn = ttk.Button(form_frame, text="Register User", 
                                      command=self.register_user, state=tk.DISABLED)
        self.register_btn.grid(row=6, column=0, columnspan=3, sticky=tk.E, pady=10)
        
        # Store captured image
        self.register_captured_image = None
        
        # Face detection indicator
        self.face_detected_label = ttk.Label(form_frame, text="", font=('Helvetica', 10))
        self.face_detected_label.grid(row=7, column=0, columnspan=3, sticky=tk.W, pady=5)

    def update_threshold_display(self, value):
        self.threshold_value.config(text=f"{int(float(value))}%")

    def init_attendance_tab(self):
        attendance_frame = ttk.Frame(self.attendance_tab)
        attendance_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(attendance_frame, text="Take Attendance", font=('Helvetica', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 20))
        
        # Create two columns
        left_frame = ttk.Frame(attendance_frame)
        left_frame.grid(row=1, column=0, sticky=tk.NSEW, padx=(0, 10))
        
        right_frame = ttk.Frame(attendance_frame)
        right_frame.grid(row=1, column=1, sticky=tk.NSEW, padx=(10, 0))
        
        attendance_frame.columnconfigure(0, weight=1)
        attendance_frame.columnconfigure(1, weight=1)
        attendance_frame.rowconfigure(1, weight=1)
        
        # Camera frame (left side)
        camera_label = ttk.Label(left_frame, text="Capture Face", font=('Helvetica', 14, 'bold'))
        camera_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Create a container for camera and buttons with proper layout
        camera_container = ttk.Frame(left_frame)
        camera_container.pack(fill=tk.BOTH, expand=True)
        
        # Camera frame with fixed height to prevent overlapping buttons
        self.attendance_camera_frame = ttk.Frame(camera_container, borderwidth=1, relief=tk.SOLID)
        self.attendance_camera_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.attendance_camera_label = ttk.Label(self.attendance_camera_frame)
        self.attendance_camera_label.pack(fill=tk.BOTH, expand=True)
        
        # Separate frame for buttons with fixed position
        camera_buttons_frame = ttk.Frame(camera_container)
        camera_buttons_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.attendance_start_camera_btn = ttk.Button(camera_buttons_frame, text="Start Camera", 
                                                    command=lambda: self.start_camera(self.attendance_camera_label))
        self.attendance_start_camera_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.attendance_capture_btn = ttk.Button(camera_buttons_frame, text="Capture Attendance", 
                                               command=self.capture_attendance_image)
        self.attendance_capture_btn.pack(side=tk.LEFT, padx=5)
        
        # Add the new "Load Image" button
        self.attendance_load_btn = ttk.Button(camera_buttons_frame, text="Load Image", 
                                            command=self.load_image_for_attendance)
        self.attendance_load_btn.pack(side=tk.LEFT, padx=5)
        
        self.attendance_reset_btn = ttk.Button(camera_buttons_frame, text="Reset", 
                                             command=self.reset_attendance, state=tk.DISABLED)
        self.attendance_reset_btn.pack(side=tk.LEFT, padx=5)
        
        # Match threshold slider
        threshold_frame = ttk.Frame(camera_container)
        threshold_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
        
        ttk.Label(threshold_frame, text="Match Threshold:").pack(side=tk.LEFT)
        self.match_threshold = ttk.Scale(threshold_frame, from_=50, to=99, orient=tk.HORIZONTAL)
        self.match_threshold.set(70)  # Default value
        self.match_threshold.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.match_threshold_value = ttk.Label(threshold_frame, text="70%")
        self.match_threshold_value.pack(side=tk.LEFT)
        
        # Update threshold value display when slider moves
        self.match_threshold.config(command=self.update_match_threshold_display)
        
        # Recognition results (right side)
        results_label = ttk.Label(right_frame, text="Recognition Results", font=('Helvetica', 14, 'bold'))
        results_label.pack(anchor=tk.W, pady=(0, 10))
        
        self.results_frame = ttk.Frame(right_frame, padding=15)
        self.results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initial message
        self.results_message = ttk.Label(self.results_frame, 
                                       text="Capture an image to see recognition results",
                                       font=('Helvetica', 12))
        self.results_message.pack(pady=20)
        
        # Store captured image
        self.attendance_captured_image = None

    def update_match_threshold_display(self, value):
        self.match_threshold_value.config(text=f"{int(float(value))}%")

    def init_users_tab(self):
        users_frame = ttk.Frame(self.users_tab)
        users_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title and search bar
        top_frame = ttk.Frame(users_frame)
        top_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = ttk.Label(top_frame, text="Manage Users", font=('Helvetica', 16, 'bold'))
        title_label.pack(side=tk.LEFT)
        
        search_frame = ttk.Frame(top_frame)
        search_frame.pack(side=tk.RIGHT)
        
        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT, padx=(0, 5))
        self.user_search_entry = ttk.Entry(search_frame, width=20)
        self.user_search_entry.pack(side=tk.LEFT)
        self.user_search_entry.bind("<KeyRelease>", self.search_users)
        
        # Users table
        table_frame = ttk.Frame(users_frame)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create Treeview
        columns = ("user_id", "name", "email", "department", "registered_on")
        self.users_table = ttk.Treeview(table_frame, columns=columns, show="headings")
        
        # Define headings
        self.users_table.heading("user_id", text="User ID")
        self.users_table.heading("name", text="Name")
        self.users_table.heading("email", text="Email")
        self.users_table.heading("department", text="Department")
        self.users_table.heading("registered_on", text="Registered On")
        
        # Define columns
        self.users_table.column("user_id", width=100)
        self.users_table.column("name", width=150)
        self.users_table.column("email", width=200)
        self.users_table.column("department", width=100)
        self.users_table.column("registered_on", width=150)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.users_table.yview)
        self.users_table.configure(yscroll=scrollbar.set)
        
        # Pack table and scrollbar
        self.users_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Buttons frame
        buttons_frame = ttk.Frame(users_frame)
        buttons_frame.pack(fill=tk.X, pady=10)
        
        refresh_btn = ttk.Button(buttons_frame, text="Refresh", command=self.load_users)
        refresh_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        delete_btn = ttk.Button(buttons_frame, text="Delete Selected", command=self.delete_user)
        delete_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Add a new button to list faces in the collection
        list_faces_btn = ttk.Button(buttons_frame, text="List Faces in Collection", 
                                   command=self.list_faces_in_collection)
        list_faces_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Add a new button to verify a user's face
        verify_btn = ttk.Button(buttons_frame, text="Verify Selected User", 
                               command=self.verify_selected_user)
        verify_btn.pack(side=tk.LEFT)
        
        # Load users
        self.load_users()

    def init_reports_tab(self):
        reports_frame = ttk.Frame(self.reports_tab)
        reports_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title and date selector
        top_frame = ttk.Frame(reports_frame)
        top_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = ttk.Label(top_frame, text="Attendance Reports", font=('Helvetica', 16, 'bold'))
        title_label.pack(side=tk.LEFT)
        
        date_frame = ttk.Frame(top_frame)
        date_frame.pack(side=tk.RIGHT)
        
        ttk.Label(date_frame, text="Date:").pack(side=tk.LEFT, padx=(0, 5))
        
        # Date entry (simplified - in real app would use a date picker)
        today = datetime.now().strftime("%Y-%m-%d")
        self.report_date_var = tk.StringVar(value=today)
        self.report_date_entry = ttk.Entry(date_frame, textvariable=self.report_date_var, width=12)
        self.report_date_entry.pack(side=tk.LEFT, padx=(0, 5))
        
        load_btn = ttk.Button(date_frame, text="Load", command=self.load_attendance_report)
        load_btn.pack(side=tk.LEFT)
        
        # Stats cards
        stats_frame = ttk.Frame(reports_frame)
        stats_frame.pack(fill=tk.X, pady=10)
        
        # Create 3 stat cards in a row
        self.report_present_card = self.create_stat_card(stats_frame, "Present", "0", 0)
        self.report_late_card = self.create_stat_card(stats_frame, "Late", "0", 1)
        self.report_absent_card = self.create_stat_card(stats_frame, "Absent", "0", 2)
        
        # Attendance table
        table_frame = ttk.Frame(reports_frame)
        table_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create Treeview
        columns = ("user_id", "name", "time", "status", "confidence")
        self.attendance_table = ttk.Treeview(table_frame, columns=columns, show="headings")
        
        # Define headings
        self.attendance_table.heading("user_id", text="User ID")
        self.attendance_table.heading("name", text="Name")
        self.attendance_table.heading("time", text="Time")
        self.attendance_table.heading("status", text="Status")
        self.attendance_table.heading("confidence", text="Confidence")
        
        # Define columns
        self.attendance_table.column("user_id", width=100)
        self.attendance_table.column("name", width=150)
        self.attendance_table.column("time", width=100)
        self.attendance_table.column("status", width=100)
        self.attendance_table.column("confidence", width=100)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.attendance_table.yview)
        self.attendance_table.configure(yscroll=scrollbar.set)
        
        # Pack table and scrollbar
        self.attendance_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Export button
        export_btn = ttk.Button(reports_frame, text="Export to CSV", command=self.export_attendance_report)
        export_btn.pack(anchor=tk.E, pady=10)
        
        # Load initial report
        self.load_attendance_report()

    def init_verification_tab(self):
        verification_frame = ttk.Frame(self.verification_tab)
        verification_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(verification_frame, text="Face Verification", font=('Helvetica', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 20))
        
        # Create two columns
        left_frame = ttk.Frame(verification_frame)
        left_frame.grid(row=1, column=0, sticky=tk.NSEW, padx=(0, 10))
        
        right_frame = ttk.Frame(verification_frame)
        right_frame.grid(row=1, column=1, sticky=tk.NSEW, padx=(10, 0))
        
        verification_frame.columnconfigure(0, weight=1)
        verification_frame.columnconfigure(1, weight=1)
        verification_frame.rowconfigure(1, weight=1)
        
        # Camera frame (left side)
        camera_label = ttk.Label(left_frame, text="Capture Face", font=('Helvetica', 14, 'bold'))
        camera_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Create a container for camera and buttons with proper layout
        camera_container = ttk.Frame(left_frame)
        camera_container.pack(fill=tk.BOTH, expand=True)
        
        # Camera frame with fixed height to prevent overlapping buttons
        self.verify_camera_frame = ttk.Frame(camera_container, borderwidth=1, relief=tk.SOLID)
        self.verify_camera_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.verify_camera_label = ttk.Label(self.verify_camera_frame)
        self.verify_camera_label.pack(fill=tk.BOTH, expand=True)
        
        # Separate frame for buttons with fixed position
        camera_buttons_frame = ttk.Frame(camera_container)
        camera_buttons_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.verify_start_camera_btn = ttk.Button(camera_buttons_frame, text="Start Camera", 
                                                command=lambda: self.start_camera(self.verify_camera_label))
        self.verify_start_camera_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.verify_capture_btn = ttk.Button(camera_buttons_frame, text="Capture Image", 
                                           command=self.capture_verify_image)
        self.verify_capture_btn.pack(side=tk.LEFT, padx=5)
        
        # Add the new "Load Image" button
        self.verify_load_btn = ttk.Button(camera_buttons_frame, text="Load Image", 
                                        command=self.load_image_for_verification)
        self.verify_load_btn.pack(side=tk.LEFT, padx=5)
        
        self.verify_reset_btn = ttk.Button(camera_buttons_frame, text="Reset", 
                                         command=self.reset_verify_image, state=tk.DISABLED)
        self.verify_reset_btn.pack(side=tk.LEFT, padx=5)
        
        # User selection (right side)
        user_label = ttk.Label(right_frame, text="Select User to Verify", font=('Helvetica', 14, 'bold'))
        user_label.pack(anchor=tk.W, pady=(0, 10))
        
        # User selection frame
        user_frame = ttk.Frame(right_frame, padding=15)
        user_frame.pack(fill=tk.BOTH, expand=True)
        
        # User ID dropdown
        ttk.Label(user_frame, text="User ID:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.verify_user_id = ttk.Combobox(user_frame, width=30, state="readonly")
        self.verify_user_id.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # Refresh user list button
        refresh_btn = ttk.Button(user_frame, text="Refresh Users", command=self.load_verify_users)
        refresh_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # Verification button
        self.verify_btn = ttk.Button(user_frame, text="Verify Face", 
                                   command=self.verify_face, state=tk.DISABLED)
        self.verify_btn.grid(row=1, column=0, columnspan=3, sticky=tk.E, pady=10)
        
        # Results frame
        results_frame = ttk.Frame(user_frame)
        results_frame.grid(row=2, column=0, columnspan=3, sticky=tk.NSEW, pady=10)
        
        # Results label
        self.verify_results = ttk.Label(results_frame, text="", font=('Helvetica', 12))
        self.verify_results.pack(pady=10)
        
        # Store captured image
        self.verify_captured_image = None
        
        # Load users for verification
        self.load_verify_users()

    def load_verify_users(self):
        try:
            users_table = dynamodb.Table(USERS_TABLE)
            response = users_table.scan()
            
            user_ids = [user.get('user_id', '') for user in response.get('Items', [])]
            user_ids.sort()
            
            self.verify_user_id['values'] = user_ids
            
            if user_ids:
                self.verify_user_id.current(0)
        except Exception as e:
            # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
            if self.debug_mode_var.get():
                print(f"Error loading users for verification: {e}")
                import traceback
                traceback.print_exc()

    # New function to load an image for registration
    def load_image_for_registration(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if not file_path:
            return  # User cancelled
        
        try:
            # Read the image
            image = cv2.imread(file_path)
            if image is None:
                messagebox.showerror("Error", "Could not read the selected image")
                return
                
            # Store the image for registration
            self.register_captured_image = image
            
            # Convert to RGB for display
            rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            
            # Resize to fit display
            display_width = self.register_camera_label.winfo_width()
            display_height = self.register_camera_label.winfo_height()
            
            if display_width > 1 and display_height > 1:
                img = self.resize_image(img, display_width, display_height)
            
            imgtk = ImageTk.PhotoImage(image=img)
            self.register_camera_label.imgtk = imgtk
            self.register_camera_label.config(image=imgtk)
            
            # Update UI
            self.register_reset_btn.config(state=tk.NORMAL)
            self.register_btn.config(state=tk.NORMAL)
            
            # Check for faces in the image
            self.check_face_in_image(self.register_captured_image, self.face_detected_label)
            
            self.register_status.config(text="Image loaded. Please fill in user details and click Register.")
            # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
            if self.debug_mode_var.get():
                print("Image loaded successfully for registration")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")
            # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
            if self.debug_mode_var.get():
                print(f"Error loading image: {e}")
                import traceback
                traceback.print_exc()

    # New function to load an image for attendance
    def load_image_for_attendance(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if not file_path:
            return  # User cancelled
        
        try:
            # Read the image
            image = cv2.imread(file_path)
            if image is None:
                messagebox.showerror("Error", "Could not read the selected image")
                return
                
            # Store the image for attendance
            self.attendance_captured_image = image
            
            # Convert to RGB for display
            rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            
            # Resize to fit display
            display_width = self.attendance_camera_label.winfo_width()
            display_height = self.attendance_camera_label.winfo_height()
            
            if display_width > 1 and display_height > 1:
                img = self.resize_image(img, display_width, display_height)
            
            imgtk = ImageTk.PhotoImage(image=img)
            self.attendance_camera_label.imgtk = imgtk
            self.attendance_camera_label.config(image=imgtk)
            
            # Update UI
            self.attendance_reset_btn.config(state=tk.NORMAL)
            
            # Process attendance
            self.process_attendance()
            # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
            if self.debug_mode_var.get():
                print("Image loaded successfully for attendance")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")
            # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
            if self.debug_mode_var.get():
                print(f"Error loading image: {e}")
                import traceback
                traceback.print_exc()

    # New function to load an image for verification
    def load_image_for_verification(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if not file_path:
            return  # User cancelled
        
        try:
            # Read the image
            image = cv2.imread(file_path)
            if image is None:
                messagebox.showerror("Error", "Could not read the selected image")
                return
                
            # Store the image for verification
            self.verify_captured_image = image
            
            # Convert to RGB for display
            rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            
            # Resize to fit display
            display_width = self.verify_camera_label.winfo_width()
            display_height = self.verify_camera_label.winfo_height()
            
            if display_width > 1 and display_height > 1:
                img = self.resize_image(img, display_width, display_height)
            
            imgtk = ImageTk.PhotoImage(image=img)
            self.verify_camera_label.imgtk = imgtk
            self.verify_camera_label.config(image=imgtk)
            
            # Update UI
            self.verify_reset_btn.config(state=tk.NORMAL)
            self.verify_btn.config(state=tk.NORMAL)
            
            # Check for faces in the image
            faces_detected = self.detect_faces_in_image(self.verify_captured_image)
            if faces_detected:
                self.verify_results.config(text="Face detected in image. Ready to verify.")
            else:
                self.verify_results.config(text="Warning: No face detected in the image.")
            
            # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
            if self.debug_mode_var.get():
                print("Image loaded successfully for verification")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")
            # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
            if self.debug_mode_var.get():
                print(f"Error loading image: {e}")
                import traceback
                traceback.print_exc()

    # New function to list faces in the collection
    def list_faces_in_collection(self):
        try:
            response = rekognition.list_faces(
                CollectionId=COLLECTION_ID,
                MaxResults=1000
            )
            
            if not response['Faces']:
                messagebox.showinfo("Info", "No faces found in the collection")
                return
            
            # Create a simple dialog to show the faces
            faces_window = tk.Toplevel(self)
            faces_window.title("Registered Faces")
            faces_window.geometry("600x400")
            
            # Create a text widget to display the faces
            text_widget = tk.Text(faces_window, wrap=tk.WORD)
            text_widget.pack(fill=tk.BOTH, expand=True)
            
            # Add a scrollbar
            scrollbar = ttk.Scrollbar(faces_window, command=text_widget.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            text_widget.config(yscrollcommand=scrollbar.set)
            
            # Display the faces
            text_widget.insert(tk.END, f"Total faces in collection: {len(response['Faces'])}\n\n")
            
            for face in response['Faces']:
                user_id = face.get('ExternalImageId', 'Unknown')
                face_id = face.get('FaceId', 'Unknown')
                confidence = face.get('Confidence', 0)
                
                # Convert confidence to Decimal for display
                confidence_decimal = Decimal(str(confidence)) if isinstance(confidence, float) else confidence
                
                text_widget.insert(tk.END, f"User ID: {user_id}\n")
                text_widget.insert(tk.END, f"Face ID: {face_id}\n")
                text_widget.insert(tk.END, f"Confidence: {float(confidence_decimal):.2f}%\n\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to list faces: {e}")
            # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
            if self.debug_mode_var.get():
                print(f"Error listing faces: {e}")
                import traceback
                traceback.print_exc()

    # New function to verify a selected user
    def verify_selected_user(self):
        selected_item = self.users_table.selection()
        if not selected_item:
            messagebox.showinfo("Info", "Please select a user to verify")
            return
        
        user_id = self.users_table.item(selected_item, 'values')[0]
        
        # Switch to verification tab and set the user
        self.notebook.select(self.verification_tab)
        
        # Set the user in the combobox
        if user_id in self.verify_user_id['values']:
            self.verify_user_id.set(user_id)
        
        # Show message to capture or load an image
        self.verify_results.config(text=f"Selected user: {user_id}. Please capture or load an image to verify.")

    # Camera functions
    def start_camera(self, display_label):
        if self.is_camera_running:
            return
        
        try:
            # Try camera index 0 first, then 1 if that fails
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
                if self.debug_mode_var.get():
                    print("Camera index 0 failed, trying index 1...")
                self.camera = cv2.VideoCapture(1)
                
            if not self.camera.isOpened():
                messagebox.showerror("Error", "Could not open camera. Please check your camera connection.")
                return
            
            # Read a test frame to confirm camera is working
            ret, test_frame = self.camera.read()
            if not ret:
                messagebox.showerror("Error", "Camera opened but could not read frames. Please check your camera.")
                self.camera.release()
                self.camera = None
                return
                
            # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
            if self.debug_mode_var.get():
                print(f"Camera initialized successfully. Frame size: {test_frame.shape}")
            
            self.is_camera_running = True
            
            # Update UI based on which tab is active
            if display_label == self.register_camera_label:
                self.register_start_camera_btn.config(text="Stop Camera", 
                                                    command=lambda: self.stop_camera(self.register_camera_label))
                # Always enable the capture button when camera starts successfully
                self.register_capture_btn.config(state=tk.NORMAL)
                # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
                if self.debug_mode_var.get():
                    print("Register capture button enabled")
            elif display_label == self.attendance_camera_label:
                self.attendance_start_camera_btn.config(text="Stop Camera", 
                                                      command=lambda: self.stop_camera(self.attendance_camera_label))
                # Always enable the capture button when camera starts successfully
                self.attendance_capture_btn.config(state=tk.NORMAL)
                # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
                if self.debug_mode_var.get():
                    print("Attendance capture button enabled")
            else:  # Verification tab
                self.verify_start_camera_btn.config(text="Stop Camera", 
                                                  command=lambda: self.stop_camera(self.verify_camera_label))
                # Always enable the capture button when camera starts successfully
                self.verify_capture_btn.config(state=tk.NORMAL)
                # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
                if self.debug_mode_var.get():
                    print("Verification capture button enabled")
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self.update_camera, args=(display_label,))
            self.camera_thread.daemon = True
            self.camera_thread.start()
        
        except Exception as e:
            messagebox.showerror("Camera Error", f"Error starting camera: {e}")
            # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
            if self.debug_mode_var.get():
                print(f"Camera error details: {e}")
                import traceback
                traceback.print_exc()
            if self.camera:
                self.camera.release()
                self.camera = None

    def stop_camera(self, display_label):
        self.is_camera_running = False
        
        # Wait for camera thread to finish
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1.0)
            
        if self.camera:
            self.camera.release()
            self.camera = None
        
        # Update UI based on which tab is active
        if display_label == self.register_camera_label:
            self.register_start_camera_btn.config(text="Start Camera", 
                                                command=lambda: self.start_camera(self.register_camera_label))
            # Disable capture button when camera is stopped
            self.register_capture_btn.config(state=tk.DISABLED)
        elif display_label == self.attendance_camera_label:
            self.attendance_start_camera_btn.config(text="Start Camera", 
                                                  command=lambda: self.start_camera(self.attendance_camera_label))
            # Disable capture button when camera is stopped
            self.attendance_capture_btn.config(state=tk.DISABLED)
        else:  # Verification tab
            self.verify_start_camera_btn.config(text="Start Camera", 
                                              command=lambda: self.start_camera(self.verify_camera_label))
            # Disable capture button when camera is stopped
            self.verify_capture_btn.config(state=tk.DISABLED)

    def update_camera(self, display_label):
        frame_count = 0
        last_time = time.time()
        
        while self.is_camera_running and self.camera:
            try:
                ret, frame = self.camera.read()
                if ret:
                    self.current_frame = frame.copy()
                    
                    # Convert to RGB for tkinter
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb_frame)
                    
                    # Resize to fit display
                    display_width = display_label.winfo_width()
                    display_height = display_label.winfo_height()
                    
                    if display_width > 1 and display_height > 1:
                        img = self.resize_image(img, display_width, display_height)
                    
                    imgtk = ImageTk.PhotoImage(image=img)
                    
                    # Update UI in the main thread
                    self.after_idle(lambda l=display_label, i=imgtk: self.update_camera_display(l, i))
                    
                    # Detect faces in the frame for the register tab
                    if display_label == self.register_camera_label and frame_count % 10 == 0:  # Check every 10 frames
                        self.after_idle(lambda f=frame: self.check_face_in_image(f, self.face_detected_label))
                    
                    # Calculate and print FPS every 30 frames
                    frame_count += 1
                    if frame_count % 30 == 0 and self.debug_mode_var.get():  # FIX: Use self.debug_mode_var.get()
                        current_time = time.time()
                        fps = 30 / (current_time - last_time)
                        last_time = current_time
                        print(f"Camera FPS: {fps:.1f}")
                else:
                    # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
                    if self.debug_mode_var.get():
                        print("Failed to read frame from camera")
                    break
                
                # Sleep to control frame rate
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
                if self.debug_mode_var.get():
                    print(f"Error in camera thread: {e}")
                break
        
        # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
        if self.debug_mode_var.get():
            print("Camera thread stopped")

    def update_camera_display(self, display_label, imgtk):
        """Update camera display in the main thread to avoid tkinter threading issues"""
        display_label.imgtk = imgtk
        display_label.config(image=imgtk)

    # Modified resize function to maintain aspect ratio but limit height
    def resize_image(self, img, target_width, target_height):
        width, height = img.size
        
        # Calculate ratio to maintain aspect ratio
        ratio = min(target_width/width, target_height/height)
        
        # Limit the height to leave room for buttons
        max_height = target_height - 50  # Leave 50 pixels for buttons
        if height * ratio > max_height:
            ratio = max_height / height
        
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        return img.resize((new_width, new_height), Image.LANCZOS)

    # Function to detect faces in an image
    def detect_faces_in_image(self, image):
        try:
            # Convert image to bytes for AWS
            _, img_encoded = cv2.imencode('.jpg', image)
            img_bytes = img_encoded.tobytes()
            
            # Detect faces with AWS Rekognition
            response = rekognition.detect_faces(
                Image={'Bytes': img_bytes},
                Attributes=['ALL']
            )
            
            return response.get('FaceDetails', [])
        except Exception as e:
            # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
            if self.debug_mode_var.get():
                print(f"Error detecting faces: {e}")
                import traceback
                traceback.print_exc()
            return []

    # Function to check for faces in an image and update UI
    def check_face_in_image(self, image, label_widget):
        faces = self.detect_faces_in_image(image)
        
        if not faces:
            label_widget.config(text="No face detected", foreground="red")
            return False
        elif len(faces) > 1:
            label_widget.config(text=f"Multiple faces detected ({len(faces)})", foreground="orange")
            return True
        else:
            quality = faces[0].get('Quality', {})
            brightness = quality.get('Brightness', 0)
            sharpness = quality.get('Sharpness', 0)
            
            # Convert to Decimal if they're floats
            brightness = Decimal(str(brightness)) if isinstance(brightness, float) else brightness
            sharpness = Decimal(str(sharpness)) if isinstance(sharpness, float) else sharpness
            
            if float(brightness) < 30 or float(sharpness) < 30:
                label_widget.config(text=f"Low quality face (Brightness: {float(brightness):.1f}, Sharpness: {float(sharpness):.1f})", 
                                  foreground="orange")
            else:
                label_widget.config(text=f"Face detected (Brightness: {float(brightness):.1f}, Sharpness: {float(sharpness):.1f})", 
                                  foreground="green")
            return True

    def capture_register_image(self):
        if self.current_frame is not None:
            self.register_captured_image = self.current_frame.copy()
            
            # Convert to RGB for display
            rgb_frame = cv2.cvtColor(self.register_captured_image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            
            # Resize to fit display
            display_width = self.register_camera_label.winfo_width()
            display_height = self.register_camera_label.winfo_height()
            
            if display_width > 1 and display_height > 1:
                img = self.resize_image(img, display_width, display_height)
            
            imgtk = ImageTk.PhotoImage(image=img)
            self.register_camera_label.imgtk = imgtk
            self.register_camera_label.config(image=imgtk)
            
            # Stop camera and update UI
            self.stop_camera(self.register_camera_label)
            self.register_reset_btn.config(state=tk.NORMAL)
            self.register_btn.config(state=tk.NORMAL)
            
            # Check for faces in the image
            self.check_face_in_image(self.register_captured_image, self.face_detected_label)
            
            self.register_status.config(text="Image captured. Please fill in user details and click Register.")
            # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
            if self.debug_mode_var.get():
                print("Register image captured successfully")
        else:
            messagebox.showerror("Error", "No camera frame available. Please make sure the camera is running.")

    def reset_register_image(self):
        self.register_captured_image = None
        self.register_camera_label.config(image="")
        self.register_reset_btn.config(state=tk.DISABLED)
        self.register_btn.config(state=tk.DISABLED)
        self.register_status.config(text="")
        self.face_detected_label.config(text="")

    def capture_attendance_image(self):
        if self.current_frame is not None:
            self.attendance_captured_image = self.current_frame.copy()
            
            # Convert to RGB for display
            rgb_frame = cv2.cvtColor(self.attendance_captured_image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            
            # Resize to fit display
            display_width = self.attendance_camera_label.winfo_width()
            display_height = self.attendance_camera_label.winfo_height()
            
            if display_width > 1 and display_height > 1:
                img = self.resize_image(img, display_width, display_height)
            
            imgtk = ImageTk.PhotoImage(image=img)
            self.attendance_camera_label.imgtk = imgtk
            self.attendance_camera_label.config(image=imgtk)
            
            # Stop camera and update UI
            self.stop_camera(self.attendance_camera_label)
            self.attendance_reset_btn.config(state=tk.NORMAL)
            
            # Process attendance
            self.process_attendance()
            # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
            if self.debug_mode_var.get():
                print("Attendance image captured successfully")
        else:
            messagebox.showerror("Error", "No camera frame available. Please make sure the camera is running.")

    def capture_verify_image(self):
        if self.current_frame is not None:
            self.verify_captured_image = self.current_frame.copy()
            
            # Convert to RGB for display
            rgb_frame = cv2.cvtColor(self.verify_captured_image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            
            # Resize to fit display
            display_width = self.verify_camera_label.winfo_width()
            display_height = self.verify_camera_label.winfo_height()
            
            if display_width > 1 and display_height > 1:
                img = self.resize_image(img, display_width, display_height)
            
            imgtk = ImageTk.PhotoImage(image=img)
            self.verify_camera_label.imgtk = imgtk
            self.verify_camera_label.config(image=imgtk)
            
            # Stop camera and update UI
            self.stop_camera(self.verify_camera_label)
            self.verify_reset_btn.config(state=tk.NORMAL)
            self.verify_btn.config(state=tk.NORMAL)
            
            # Check for faces in the image
            faces_detected = self.detect_faces_in_image(self.verify_captured_image)
            if faces_detected:
                self.verify_results.config(text="Face detected in image. Ready to verify.")
            else:
                self.verify_results.config(text="Warning: No face detected in the image.")
            
            # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
            if self.debug_mode_var.get():
                print("Verification image captured successfully")
        else:
            messagebox.showerror("Error", "No camera frame available. Please make sure the camera is running.")

    def reset_attendance(self):
        self.attendance_captured_image = None
        self.attendance_camera_label.config(image="")
        self.attendance_reset_btn.config(state=tk.DISABLED)
        
        # Clear results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        self.results_message = ttk.Label(self.results_frame, 
                                       text="Capture an image to see recognition results",
                                       font=('Helvetica', 12))
        self.results_message.pack(pady=20)

    def reset_verify_image(self):
        self.verify_captured_image = None
        self.verify_camera_label.config(image="")
        self.verify_reset_btn.config(state=tk.DISABLED)
        self.verify_btn.config(state=tk.DISABLED)
        self.verify_results.config(text="")

    # User registration with enhanced debugging and Decimal conversion
    def register_user(self):
        # Validate inputs
        user_id = self.register_user_id.get().strip()
        name = self.register_name.get().strip()
        email = self.register_email.get().strip()
        department = self.register_department.get().strip()
        
        if not user_id or not name or not email or not department:
            self.register_status.config(text="Please fill in all fields")
            return
        
        if self.register_captured_image is None:
            self.register_status.config(text="Please capture an image first")
            return
        
        try:
            # Convert image to bytes for AWS
            _, img_encoded = cv2.imencode('.jpg', self.register_captured_image)
            img_bytes = img_encoded.tobytes()
            
            # Get quality threshold from slider
            quality_threshold = int(float(self.quality_threshold.get()))
            
            # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
            if self.debug_mode_var.get():
                print(f"Sending image to AWS Rekognition (size: {len(img_bytes)} bytes)")
                print(f"Quality threshold: {quality_threshold}%")
            
            # Check if user already exists
            users_table = dynamodb.Table(USERS_TABLE)
            response = users_table.get_item(Key={'user_id': user_id})
            
            if 'Item' in response:
                self.register_status.config(text=f"User ID {user_id} already exists")
                return
            
            # Check face quality first
            face_details = self.detect_faces_in_image(self.register_captured_image)
            
            if not face_details:
                self.register_status.config(text="No face detected in the image. Please try again.")
                # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
                if self.debug_mode_var.get():
                    print("No faces detected in the image")
                return
            
            if len(face_details) > 1:
                self.register_status.config(text="Multiple faces detected. Please use an image with only one face.")
                # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
                if self.debug_mode_var.get():
                    print(f"Multiple faces detected: {len(face_details)}")
                return
            
            # Check face quality
            quality = face_details[0].get('Quality', {})
            brightness = quality.get('Brightness', 0)
            sharpness = quality.get('Sharpness', 0)
            
            # Convert to Decimal if they're floats
            brightness = Decimal(str(brightness)) if isinstance(brightness, float) else brightness
            sharpness = Decimal(str(sharpness)) if isinstance(sharpness, float) else sharpness
            
            if float(brightness) < 30 or float(sharpness) < 30:
                if not messagebox.askyesno("Low Quality Face", 
                                         f"The face image has low quality (Brightness: {float(brightness):.1f}, Sharpness: {float(sharpness):.1f}).\nThis may affect recognition accuracy. Continue anyway?"):
                    return
            
            # FIX: Correct the QualityFilter format - use one of the allowed values: AUTO, LOW, MEDIUM, HIGH, NONE
            # Instead of "AUTO:{quality_threshold}", we'll map the threshold to the appropriate quality level
            quality_filter = "AUTO"  # Default value
            
            # Map the threshold to the appropriate quality level
            if quality_threshold >= 90:
                quality_filter = "HIGH"
            elif quality_threshold >= 70:
                quality_filter = "MEDIUM"
            elif quality_threshold >= 50:
                quality_filter = "LOW"
            else:
                quality_filter = "NONE"
            
            # Register face with AWS Rekognition
            rekognition_response = rekognition.index_faces(
                CollectionId=COLLECTION_ID,
                Image={'Bytes': img_bytes},
                ExternalImageId=user_id,
                DetectionAttributes=['ALL'],
                QualityFilter=quality_filter  # Use the mapped quality filter value
            )
            
            # Add detailed debug information
            # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
            if self.debug_mode_var.get():
                print(f"Rekognition response: {json.dumps(rekognition_response, indent=2)}")
            
            if not rekognition_response['FaceRecords']:
                self.register_status.config(text="No face detected or face quality too low. Please try again.")
                # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
                if self.debug_mode_var.get():
                    print("No faces indexed by Rekognition")
                return
            
            face_id = rekognition_response['FaceRecords'][0]['Face']['FaceId']
            confidence = rekognition_response['FaceRecords'][0]['Face']['Confidence']
            
            # Convert confidence to Decimal for DynamoDB
            confidence_decimal = Decimal(str(confidence))
            
            # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
            if self.debug_mode_var.get():
                print(f"Face detected with confidence: {float(confidence_decimal)}%, Face ID: {face_id}")
            
            # Store user in DynamoDB
            timestamp = datetime.now().isoformat()
            users_table.put_item(
                Item={
                    'user_id': user_id,
                    'name': name,
                    'email': email,
                    'department': department,
                    'face_id': face_id,
                    'confidence': confidence_decimal,  # Store as Decimal
                    'created_at': timestamp
                }
            )
            
            # Clear form and show success message
            self.register_user_id.delete(0, tk.END)
            self.register_name.delete(0, tk.END)
            self.register_email.delete(0, tk.END)
            self.register_department.set("")
            self.reset_register_image()
            
            messagebox.showinfo("Success", f"User {name} registered successfully!")
            
            # Refresh users list
            self.load_users()
            
            # Update dashboard stats
            self.update_dashboard_stats()
            
        except Exception as e:
            self.register_status.config(text=f"Error: {e}")
            # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
            if self.debug_mode_var.get():
                print(f"Registration error details: {e}")
                import traceback
                traceback.print_exc()

    # Verify face against a specific user
    def verify_face(self):
        if self.verify_captured_image is None:
            messagebox.showinfo("Info", "Please capture or load an image first")
            return
        
        user_id = self.verify_user_id.get()
        if not user_id:
            messagebox.showinfo("Info", "Please select a user to verify against")
            return
        
        try:
            # Convert image to bytes for AWS
            _, img_encoded = cv2.imencode('.jpg', self.verify_captured_image)
            img_bytes = img_encoded.tobytes()
            
            # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
            if self.debug_mode_var.get():
                print(f"Sending image to AWS Rekognition for verification (size: {len(img_bytes)} bytes)")
            
            # First, detect faces to ensure there's a face in the image
            face_details = self.detect_faces_in_image(self.verify_captured_image)
            
            if not face_details:
                self.verify_results.config(text="No face detected in the image. Please try again.")
                return
            
            if len(face_details) > 1:
                self.verify_results.config(text="Multiple faces detected. Please use an image with only one face.")
                return
            
            # Search for face in Rekognition
            search_response = rekognition.search_faces_by_image(
                CollectionId=COLLECTION_ID,
                Image={'Bytes': img_bytes},
                MaxFaces=10,
                FaceMatchThreshold=50  # Lower threshold to get more potential matches
            )
            
            # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
            if self.debug_mode_var.get():
                print(f"Search response: {json.dumps(search_response, indent=2)}")
            
            # Check if the user is in the matches
            user_match = None
            for match in search_response.get('FaceMatches', []):
                if match['Face']['ExternalImageId'] == user_id:
                    user_match = match
                    break
            
            if user_match:
                confidence = user_match['Similarity']
                confidence_decimal = Decimal(str(confidence)) if isinstance(confidence, float) else confidence
                
                # Get user details
                users_table = dynamodb.Table(USERS_TABLE)
                user_response = users_table.get_item(Key={'user_id': user_id})
                
                if 'Item' in user_response:
                    user = user_response['Item']
                    name = user.get('name', 'Unknown')
                    
                    # Display verification result
                    if float(confidence_decimal) >= 80:
                        self.verify_results.config(
                            text=f"✅ VERIFIED: {name} ({user_id})\nConfidence: {float(confidence_decimal):.2f}%",
                            foreground="green"
                        )
                    else:
                        self.verify_results.config(
                            text=f"⚠️ POSSIBLE MATCH: {name} ({user_id})\nConfidence: {float(confidence_decimal):.2f}%",
                            foreground="orange"
                        )
                else:
                    self.verify_results.config(
                        text=f"⚠️ User data not found for {user_id}\nMatch confidence: {float(confidence_decimal):.2f}%",
                        foreground="orange"
                    )
            else:
                # No match found for this user
                self.verify_results.config(
                    text=f"❌ NOT VERIFIED: No match found for {user_id}",
                    foreground="red"
                )
                
                # Check if there were any other matches
                if search_response.get('FaceMatches'):
                    other_match = search_response['FaceMatches'][0]
                    other_id = other_match['Face']['ExternalImageId']
                    other_confidence = other_match['Similarity']
                    other_confidence_decimal = Decimal(str(other_confidence)) if isinstance(other_confidence, float) else other_confidence
                    
                    self.verify_results.config(
                        text=f"❌ NOT VERIFIED: No match found for {user_id}\n\nPossible match with {other_id} ({float(other_confidence_decimal):.2f}%)",
                        foreground="red"
                    )
        
        except Exception as e:
            self.verify_results.config(text=f"Error: {e}", foreground="red")
            # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
            if self.debug_mode_var.get():
                print(f"Verification error: {e}")
                import traceback
                traceback.print_exc()

    # Attendance processing with Decimal conversion
    def process_attendance(self):
        if self.attendance_captured_image is None:
            return
        
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        processing_label = ttk.Label(self.results_frame, text="Processing...", font=('Helvetica', 12))
        processing_label.pack(pady=20)
        self.update()
        
        try:
            # Convert image to bytes for AWS
            _, img_encoded = cv2.imencode('.jpg', self.attendance_captured_image)
            img_bytes = img_encoded.tobytes()
            
            # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
            if self.debug_mode_var.get():
                print(f"Sending image to AWS Rekognition for face search (size: {len(img_bytes)} bytes)")
            
            # Get match threshold from slider
            match_threshold = int(float(self.match_threshold.get()))
            
            # First, detect faces to ensure there's a face in the image
            face_details = self.detect_faces_in_image(self.attendance_captured_image)
            
            if not face_details:
                # Clear processing message
                processing_label.destroy()
                
                result_label = ttk.Label(self.results_frame, 
                                       text="No face detected in the image",
                                       font=('Helvetica', 14, 'bold'))
                result_label.pack(pady=10)
                
                message_label = ttk.Label(self.results_frame, 
                                        text="Please capture a clear image with a face.",
                                        font=('Helvetica', 12))
                message_label.pack(pady=5)
                
                return
            
            # Search for face in Rekognition
            search_response = rekognition.search_faces_by_image(
                CollectionId=COLLECTION_ID,
                Image={'Bytes': img_bytes},
                MaxFaces=1,
                FaceMatchThreshold=match_threshold
            )
            
            # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
            if self.debug_mode_var.get():
                print(f"Search response: {json.dumps(search_response, indent=2)}")
            
            # Clear processing message
            processing_label.destroy()
            
            if not search_response['FaceMatches']:
                # No match found
                result_label = ttk.Label(self.results_frame, 
                                       text="No matching face found",
                                       font=('Helvetica', 14, 'bold'))
                result_label.pack(pady=10)
                
                message_label = ttk.Label(self.results_frame, 
                                        text="This person is not registered in the system.",
                                        font=('Helvetica', 12))
                message_label.pack(pady=5)
                
                register_btn = ttk.Button(self.results_frame, text="Register New User", 
                                        command=lambda: self.notebook.select(self.register_tab))
                register_btn.pack(pady=10)
                
                return
            
            # Get the best match
            match = search_response['FaceMatches'][0]
            user_id = match['Face']['ExternalImageId']
            confidence = match['Similarity']
            
            # Convert confidence to Decimal for DynamoDB
            confidence_decimal = Decimal(str(confidence))
            
            # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
            if self.debug_mode_var.get():
                print(f"Match found: User ID: {user_id}, Confidence: {float(confidence_decimal)}%")
            
            # Get user details from DynamoDB
            users_table = dynamodb.Table(USERS_TABLE)
            user_response = users_table.get_item(Key={'user_id': user_id})
            
            if 'Item' not in user_response:
                # This shouldn't happen if the system is working correctly
                result_label = ttk.Label(self.results_frame, 
                                       text="Error: User data not found",
                                       font=('Helvetica', 14, 'bold'))
                result_label.pack(pady=10)
                return
            
            user = user_response['Item']
            
            # Determine attendance status (simplified for demo)
            now = datetime.now()
            hour = now.hour
            status = "present"
            
            # Example: If after 9 AM, mark as late
            if hour >= 9:
                status = "late"
            
            timestamp = now.isoformat()
            
            # Record attendance in DynamoDB
            attendance_table = dynamodb.Table(ATTENDANCE_TABLE)
            attendance_table.put_item(
                Item={
                    'user_id': user_id,
                    'name': user['name'],
                    'timestamp': timestamp,
                    'status': status,
                    'confidence': confidence_decimal  # Store as Decimal
                }
            )
            
            # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
            if self.debug_mode_var.get():
                print(f"Attendance recorded for {user['name']} with status: {status}")
            
            # Display results
            status_color = "#4CAF50" if status == "present" else "#FF9800"  # Green for present, orange for late
            
            status_frame = ttk.Frame(self.results_frame)
            status_frame.pack(fill=tk.X, pady=10)
            
            status_label = ttk.Label(status_frame, 
                                   text=status.upper(),
                                   font=('Helvetica', 16, 'bold'))
            status_label.pack(side=tk.LEFT)
            
            time_label = ttk.Label(status_frame, 
                                 text=f"at {now.strftime('%H:%M:%S')}",
                                 font=('Helvetica', 12))
            time_label.pack(side=tk.LEFT, padx=10)
            
            # User info
            info_frame = ttk.Frame(self.results_frame, padding=10)
            info_frame.pack(fill=tk.X, pady=10)
            
            ttk.Label(info_frame, text="Name:", font=('Helvetica', 11, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=2)
            ttk.Label(info_frame, text=user['name']).grid(row=0, column=1, sticky=tk.W, pady=2)
            
            ttk.Label(info_frame, text="User ID:", font=('Helvetica', 11, 'bold')).grid(row=1, column=0, sticky=tk.W, pady=2)
            ttk.Label(info_frame, text=user_id).grid(row=1, column=1, sticky=tk.W, pady=2)
            
            ttk.Label(info_frame, text="Department:", font=('Helvetica', 11, 'bold')).grid(row=2, column=0, sticky=tk.W, pady=2)
            ttk.Label(info_frame, text=user['department']).grid(row=2, column=1, sticky=tk.W, pady=2)
            
            ttk.Label(info_frame, text="Confidence:", font=('Helvetica', 11, 'bold')).grid(row=3, column=0, sticky=tk.W, pady=2)
            ttk.Label(info_frame, text=f"{float(confidence_decimal):.2f}%").grid(row=3, column=1, sticky=tk.W, pady=2)
            
            # Success message
            success_label = ttk.Label(self.results_frame, 
                                    text="Attendance recorded successfully!",
                                    font=('Helvetica', 12))
            success_label.pack(pady=10)
            
            # Another button
            another_btn = ttk.Button(self.results_frame, text="Record Another", 
                                    command=self.reset_attendance)
            another_btn.pack(pady=10)
            
            # Update dashboard stats
            self.update_dashboard_stats()
            
        except Exception as e:
            # Clear processing message
            processing_label.destroy()
            
            error_label = ttk.Label(self.results_frame, 
                                  text=f"Error processing attendance: {e}",
                                  font=('Helvetica', 12))
            error_label.pack(pady=20)
            # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
            if self.debug_mode_var.get():
                print(f"Attendance processing error: {e}")
                import traceback
                traceback.print_exc()

    # User management
    def load_users(self):
        # Clear existing items
        for item in self.users_table.get_children():
            self.users_table.delete(item)
        
        try:
            users_table = dynamodb.Table(USERS_TABLE)
            response = users_table.scan()
            
            for user in response.get('Items', []):
                # Format date
                created_at = user.get('created_at', '')
                if created_at:
                    try:
                        date_obj = datetime.fromisoformat(created_at)
                        formatted_date = date_obj.strftime("%Y-%m-%d %H:%M")
                    except:
                        formatted_date = created_at
                else:
                    formatted_date = ''
                
                self.users_table.insert('', tk.END, values=(
                    user.get('user_id', ''),
                    user.get('name', ''),
                    user.get('email', ''),
                    user.get('department', ''),
                    formatted_date
                ))
            
            # Also update the verification user dropdown
            self.load_verify_users()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load users: {e}")
            # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
            if self.debug_mode_var.get():
                print(f"Error loading users: {e}")
                import traceback
                traceback.print_exc()

    def search_users(self, event):
        search_term = self.user_search_entry.get().lower()
        
        # Clear existing items
        for item in self.users_table.get_children():
            self.users_table.delete(item)
        
        try:
            users_table = dynamodb.Table(USERS_TABLE)
            response = users_table.scan()
            
            for user in response.get('Items', []):
                # Check if search term is in any field
                if (search_term in user.get('user_id', '').lower() or
                    search_term in user.get('name', '').lower() or
                    search_term in user.get('email', '').lower() or
                    search_term in user.get('department', '').lower()):
                    
                    # Format date
                    created_at = user.get('created_at', '')
                    if created_at:
                        try:
                            date_obj = datetime.fromisoformat(created_at)
                            formatted_date = date_obj.strftime("%Y-%m-%d %H:%M")
                        except:
                            formatted_date = created_at
                    else:
                        formatted_date = ''
                    
                    self.users_table.insert('', tk.END, values=(
                        user.get('user_id', ''),
                        user.get('name', ''),
                        user.get('email', ''),
                        user.get('department', ''),
                        formatted_date
                    ))
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to search users: {e}")
            # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
            if self.debug_mode_var.get():
                print(f"Error searching users: {e}")
                import traceback
                traceback.print_exc()

    def delete_user(self):
        selected_item = self.users_table.selection()
        if not selected_item:
            messagebox.showinfo("Info", "Please select a user to delete")
            return
        
        user_id = self.users_table.item(selected_item, 'values')[0]
        
        if not messagebox.askyesno("Confirm", f"Are you sure you want to delete user {user_id}?"):
            return
        
        try:
            # Get face_id from DynamoDB
            users_table = dynamodb.Table(USERS_TABLE)
            response = users_table.get_item(Key={'user_id': user_id})
            
            if 'Item' not in response:
                messagebox.showerror("Error", f"User {user_id} not found")
                return
            
            face_id = response['Item'].get('face_id')
            
            # Delete face from Rekognition
            if face_id:
                rekognition.delete_faces(
                    CollectionId=COLLECTION_ID,
                    FaceIds=[face_id]
                )
                # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
                if self.debug_mode_var.get():
                    print(f"Deleted face {face_id} from Rekognition collection")
            
            # Delete user from DynamoDB
            users_table.delete_item(Key={'user_id': user_id})
            # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
            if self.debug_mode_var.get():
                print(f"Deleted user {user_id} from DynamoDB")
            
            # Remove from table
            self.users_table.delete(selected_item)
            
            messagebox.showinfo("Success", f"User {user_id} deleted successfully")
            
            # Update dashboard stats
            self.update_dashboard_stats()
            
            # Update verification user dropdown
            self.load_verify_users()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete user: {e}")
            # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
            if self.debug_mode_var.get():
                print(f"Error deleting user: {e}")
                import traceback
                traceback.print_exc()

    # Reports
    def load_attendance_report(self):
        # Clear existing items
        for item in self.attendance_table.get_children():
            self.attendance_table.delete(item)
        
        try:
            date_str = self.report_date_var.get()
            
            # Validate date format
            try:
                datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                messagebox.showerror("Error", "Invalid date format. Please use YYYY-MM-DD")
                return
            
            # Get attendance records for the date
            attendance_table = dynamodb.Table(ATTENDANCE_TABLE)
            response = attendance_table.scan(
                FilterExpression="begins_with(#ts, :date)",
                ExpressionAttributeNames={"#ts": "timestamp"},
                ExpressionAttributeValues={":date": date_str}
            )
            
            records = response.get('Items', [])
            
            # Get all users
            users_table = dynamodb.Table(USERS_TABLE)
            users_response = users_table.scan()
            all_users = users_response.get('Items', [])
            
            # Calculate stats
            present_users = set()
            late_users = set()
            
            for record in records:
                user_id = record.get('user_id')
                status = record.get('status')
                
                if status == 'present':
                    present_users.add(user_id)
                elif status == 'late':
                    late_users.add(user_id)
            
            # Update stats cards
            present_count = len(present_users)
            late_count = len(late_users)
            total_users = len(all_users)
            absent_count = total_users - present_count - late_count
            
            self.report_present_card.value_label.config(text=str(present_count))
            self.report_late_card.value_label.config(text=str(late_count))
            self.report_absent_card.value_label.config(text=str(absent_count))
            
            # Format and display attendance records
            for record in records:
                # Format time
                timestamp = record.get('timestamp', '')
                if timestamp:
                    try:
                        date_obj = datetime.fromisoformat(timestamp)
                        formatted_time = date_obj.strftime("%H:%M:%S")
                    except:
                        formatted_time = timestamp
                else:
                    formatted_time = ''
                
                # Convert confidence to string for display (if it's a Decimal)
                confidence = record.get('confidence', 0)
                if isinstance(confidence, Decimal):
                    confidence_display = f"{float(confidence):.2f}%"
                else:
                    confidence_display = f"{confidence:.2f}%"
                
                # Add to table
                self.attendance_table.insert('', tk.END, values=(
                    record.get('user_id', ''),
                    record.get('name', ''),
                    formatted_time,
                    record.get('status', ''),
                    confidence_display
                ))
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load attendance report: {e}")
            # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
            if self.debug_mode_var.get():
                print(f"Error loading attendance report: {e}")
                import traceback
                traceback.print_exc()

    def export_attendance_report(self):
        try:
            date_str = self.report_date_var.get()
            
            # Get attendance records for the date
            attendance_table = dynamodb.Table(ATTENDANCE_TABLE)
            response = attendance_table.scan(
                FilterExpression="begins_with(#ts, :date)",
                ExpressionAttributeNames={"#ts": "timestamp"},
                ExpressionAttributeValues={":date": date_str}
            )
            
            records = response.get('Items', [])
            
            if not records:
                messagebox.showinfo("Info", "No records to export")
                return
            
            # Ask user for save location
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                initialfile=f"attendance_{date_str}.csv"
            )
            
            if not file_path:
                return  # User cancelled
            
            # Create CSV content
            csv_content = "User ID,Name,Time,Status,Confidence\n"
            
            for record in records:
                # Format time
                timestamp = record.get('timestamp', '')
                if timestamp:
                    try:
                        date_obj = datetime.fromisoformat(timestamp)
                        formatted_time = date_obj.strftime("%H:%M:%S")
                    except:
                        formatted_time = timestamp
                else:
                    formatted_time = ''
                
                # Convert confidence to string for CSV (if it's a Decimal)
                confidence = record.get('confidence', 0)
                if isinstance(confidence, Decimal):
                    confidence_display = f"{float(confidence):.2f}%"
                else:
                    confidence_display = f"{confidence:.2f}%"
                
                csv_content += f"{record.get('user_id', '')},{record.get('name', '')},{formatted_time},"
                csv_content += f"{record.get('status', '')},{confidence_display}\n"
            
            # Save to file
            with open(file_path, 'w') as f:
                f.write(csv_content)
            
            messagebox.showinfo("Success", f"Report exported to {file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export report: {e}")
            # FIX: Use self.debug_mode_var.get() instead of self.debug_mode.get()
            if self.debug_mode_var.get():
                print(f"Error exporting report: {e}")
                import traceback
                traceback.print_exc()

    def on_closing(self):
        if self.is_camera_running:
            self.is_camera_running = False
            if self.camera:
                self.camera.release()
        
        self.destroy()

# Run the application
if __name__ == "__main__":
    app = FaceAttendanceApp()
    app.mainloop()
