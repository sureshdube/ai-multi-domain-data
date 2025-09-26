import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import Dashboard from './src/Dashboard';

// Mock API
jest.mock('./src/api', () => ({
  getTrends: jest.fn(() => Promise.resolve({ trends: [
    { id: 1, name: 'Delayed Deliveries', count: 2 },
    { id: 2, name: 'On-Time Deliveries', count: 1 },
  ] })),
  getCasesForTrend: jest.fn((id) => Promise.resolve({ cases: id === 1 ? [
    { case_id: 'D001', order_id: 'ORD123', status: 'Delayed', customer: 'Alice', details: 'Weather delay' },
    { case_id: 'D002', order_id: 'ORD124', status: 'Delayed', customer: 'Bob', details: 'Traffic jam' },
  ] : [] }))
}));

describe('Dashboard', () => {
  it('renders trends and drill-down cases', async () => {
    render(<Dashboard />);
    expect(screen.getByText(/Delivery Trends/i)).toBeInTheDocument();
    await waitFor(() => screen.getByText(/Delayed Deliveries/i));
    fireEvent.click(screen.getByText(/Delayed Deliveries/i));
    await waitFor(() => screen.getByText(/Cases for: Delayed Deliveries/i));
    expect(screen.getByText('D001')).toBeInTheDocument();
    expect(screen.getByText('Alice')).toBeInTheDocument();
  });
});
