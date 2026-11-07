"""User ORM model with role-based access control."""

import enum
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, Index, String, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from atlas_rag.database import Base


class UserRole(str, enum.Enum):
    """User roles for RBAC."""

    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"
    API_ONLY = "api_only"


class User(Base):
    """Represents a platform user with role-based permissions."""

    __tablename__ = "users"
    __table_args__ = (
        Index("ix_users_email", "email"),
        Index("ix_users_api_key", "api_key"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), default=UserRole.VIEWER, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    api_key: Mapped[str | None] = mapped_column(String(64), unique=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    documents: Mapped[list["Document"]] = relationship(  # noqa: F821
        back_populates="owner", cascade="all, delete-orphan"
    )
    collections: Mapped[list["Collection"]] = relationship(  # noqa: F821
        back_populates="owner", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User {self.email} role={self.role.value}>"
