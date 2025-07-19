"""
Data models for the network monitoring system.

This module defines Pydantic models for:
- KPI data structures
- Anomaly detection results
- Ticket information
- Email notifications
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, validator
import pandas as pd


class KPIType(str, Enum):
    """Supported KPI types."""
    TRAFFIC_CHANNEL_ASSIGNMENT = "3G Traffic Channel Assignment Success Rate"
    PAGING_SUCCESS_RATE = "Paging Success Rate"
    APN_TRAFFIC = "APN Traffic"
    INCOMING_TRUNK_OFFICE = "Incoming Trunk Office"


class AnomalyType(str, Enum):
    """Types of anomalies that can be detected."""
    UPPER_BOUND = "upper_bound"
    LOWER_BOUND = "lower_bound"
    TREND_INCREASE = "trend_increase"
    TREND_DECREASE = "trend_decrease"
    SEASONAL = "seasonal"


class TicketStatus(str, Enum):
    """Ticket status options."""
    RUNNING = "Running"
    CLOSED = "Closed"
    PENDING = "Pending"
    RESOLVED = "Resolved"


class KPIData(BaseModel):
    """Model for KPI measurement data."""
    sdate: datetime = Field(..., description="Measurement timestamp")
    snode: str = Field(..., description="Network node identifier")
    value: float = Field(..., description="KPI value")
    kpi_name: str = Field(..., description="Name of the KPI")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class LocationInfo(BaseModel):
    """Model for network node location information."""
    snode: str = Field(..., description="Network node identifier")
    city: str = Field(..., description="City name")
    province: str = Field(..., description="Province name")
    region: Optional[str] = Field(None, description="Region name")


class AnomalyPoint(BaseModel):
    """Model for individual anomaly detection points."""
    timestamp: datetime = Field(..., description="Anomaly timestamp")
    actual_value: float = Field(..., description="Actual measured value")
    predicted_value: float = Field(..., description="Predicted value")
    lower_bound: float = Field(..., description="Lower prediction bound")
    upper_bound: float = Field(..., description="Upper prediction bound")
    anomaly_type: AnomalyType = Field(..., description="Type of anomaly")
    severity: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Anomaly severity score (0-1)"
    )


class TrendAnalysis(BaseModel):
    """Model for trend analysis results."""
    slope: float = Field(..., description="Trend slope value")
    slope_percentage: float = Field(..., description="Slope as percentage")
    trend_direction: str = Field(..., description="Incremental or Decremental")
    is_significant: bool = Field(..., description="Whether trend is significant")
    trend_flag: int = Field(..., description="Trend alert flag (0 or 1)")
    color: str = Field(default="green", description="Display color for trend")


class DetectionResult(BaseModel):
    """Model for anomaly detection results."""
    snode: str = Field(..., description="Network node identifier")
    kpi_name: str = Field(..., description="KPI name")
    detection_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When detection was performed"
    )
    anomalies: List[AnomalyPoint] = Field(
        default_factory=list,
        description="List of detected anomalies"
    )
    trend_analysis: Optional[TrendAnalysis] = Field(
        None,
        description="Trend analysis results"
    )
    chart_path: Optional[str] = Field(
        None,
        description="Path to generated chart file"
    )
    has_anomalies: bool = Field(
        default=False,
        description="Whether any anomalies were detected"
    )
    
    @validator('has_anomalies', always=True)
    def set_has_anomalies(cls, v, values):
        """Set has_anomalies based on anomalies list."""
        return len(values.get('anomalies', [])) > 0


class TicketInfo(BaseModel):
    """Model for trouble ticket information."""
    ticket_id: Optional[str] = Field(None, description="Ticket ID")
    title: str = Field(..., description="Ticket title")
    description: str = Field(..., description="Ticket description")
    snode: str = Field(..., description="Network node identifier")
    kpi_name: str = Field(..., description="Related KPI name")
    status: TicketStatus = Field(
        default=TicketStatus.RUNNING,
        description="Ticket status"
    )
    creation_time: datetime = Field(
        default_factory=datetime.now,
        description="When ticket was created"
    )
    fault_first_occur_time: datetime = Field(
        ...,
        description="When the fault first occurred"
    )
    department: str = Field(default="Core Network", description="Responsible department")
    domain: str = Field(default="CS", description="Network domain")
    fault_level: str = Field(default="Major", description="Fault severity level")
    location: Optional[LocationInfo] = Field(None, description="Node location info")


class EmailNotification(BaseModel):
    """Model for email notification data."""
    subject: str = Field(..., description="Email subject")
    recipient_type: str = Field(..., description="Type of recipients")
    snode: str = Field(..., description="Network node identifier")
    kpi_name: str = Field(..., description="KPI name")
    detection_result: DetectionResult = Field(..., description="Detection results")
    ticket_info: Optional[TicketInfo] = Field(None, description="Related ticket info")
    chart_path: Optional[str] = Field(None, description="Path to chart image")
    template_type: str = Field(default="raise", description="Email template type")
    
    
class SystemMetrics(BaseModel):
    """Model for system performance metrics."""
    timestamp: datetime = Field(default_factory=datetime.now)
    total_nodes_monitored: int = Field(..., ge=0)
    kpis_processed: int = Field(..., ge=0)
    anomalies_detected: int = Field(..., ge=0)
    tickets_raised: int = Field(..., ge=0)
    emails_sent: int = Field(..., ge=0)
    processing_time_seconds: float = Field(..., ge=0.0)
    errors_count: int = Field(default=0, ge=0)


class KPIQuery(BaseModel):
    """Model for KPI data query parameters."""
    kpi_name: str = Field(..., description="Name of KPI to query")
    start_time: datetime = Field(..., description="Query start time")
    end_time: datetime = Field(..., description="Query end time")
    snode_filter: Optional[str] = Field(None, description="Filter by specific node")
    
    @validator('end_time')
    def end_time_must_be_after_start_time(cls, v, values):
        """Validate that end_time is after start_time."""
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError('end_time must be after start_time')
        return v


class DatabaseConnection(BaseModel):
    """Model for database connection information."""
    host: str
    port: int
    username: str
    password: str
    dsn: str
    encoding: str = "UTF-8"
    is_connected: bool = False
    connection_time: Optional[datetime] = None
    
    class Config:
        """Pydantic configuration."""
        # Don't include password in string representation
        repr_exclude = {'password'}


# Utility functions for data conversion

def dataframe_to_kpi_data(df: pd.DataFrame, kpi_name: str) -> List[KPIData]:
    """
    Convert pandas DataFrame to list of KPIData objects.
    
    Args:
        df: DataFrame with columns: SDATE, SNODE, and value column
        kpi_name: Name of the KPI
        
    Returns:
        List of KPIData objects
    """
    kpi_data_list = []
    
    # Determine value column (assume third column if not specified)
    value_columns = [col for col in df.columns if col not in ['SDATE', 'SNODE']]
    if not value_columns:
        raise ValueError("No value column found in DataFrame")
    
    value_column = value_columns[0]  # Use first non-date/node column
    
    for _, row in df.iterrows():
        kpi_data = KPIData(
            sdate=pd.to_datetime(row['SDATE']),
            snode=row['SNODE'],
            value=float(row[value_column]),
            kpi_name=kpi_name
        )
        kpi_data_list.append(kpi_data)
    
    return kpi_data_list


def kpi_data_to_dataframe(kpi_data_list: List[KPIData]) -> pd.DataFrame:
    """
    Convert list of KPIData objects to pandas DataFrame.
    
    Args:
        kpi_data_list: List of KPIData objects
        
    Returns:
        pandas DataFrame
    """
    data = []
    for kpi_data in kpi_data_list:
        data.append({
            'SDATE': kpi_data.sdate,
            'SNODE': kpi_data.snode,
            'value': kpi_data.value,
            'kpi_name': kpi_data.kpi_name
        })
    
    return pd.DataFrame(data)


def create_anomaly_point(
    timestamp: datetime,
    actual: float,
    predicted: float,
    lower_bound: float,
    upper_bound: float,
    sensitivity: float = 0.01
 ) -> Optional[AnomalyPoint]:
    """
    Create an AnomalyPoint if the actual value is anomalous.
    
    Args:
        timestamp: Data timestamp
        actual: Actual measured value
        predicted: Predicted value
        lower_bound: Lower prediction bound
        upper_bound: Upper prediction bound
        sensitivity: Anomaly detection sensitivity
        
    Returns:
        AnomalyPoint if anomalous, None otherwise
    """
    adjusted_lower = lower_bound - (sensitivity * lower_bound)
    adjusted_upper = upper_bound + (sensitivity * upper_bound)
    
    if actual < adjusted_lower:
        severity = abs(actual - adjusted_lower) / abs(adjusted_lower) if adjusted_lower != 0 else 1.0
        return AnomalyPoint(
            timestamp=timestamp,
            actual_value=actual,
            predicted_value=predicted,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            anomaly_type=AnomalyType.LOWER_BOUND,
            severity=min(severity, 1.0)
        )
    elif actual > adjusted_upper:
        severity = abs(actual - adjusted_upper) / abs(adjusted_upper) if adjusted_upper != 0 else 1.0
        return AnomalyPoint(
            timestamp=timestamp,
            actual_value=actual,
            predicted_value=predicted,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            anomaly_type=AnomalyType.UPPER_BOUND,
            severity=min(severity, 1.0)
        )
    
    return None 