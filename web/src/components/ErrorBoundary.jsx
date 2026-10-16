import { Component } from "react";
import { AlertTriangle, RotateCcw } from "lucide-react";

// Contains a render error to one panel instead of blanking the whole app,
// and shows the actual message so the failure can be reported/fixed.
export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    // keep the stack in the console for diagnosis
    console.error(`[${this.props.name || "panel"}] render error:`, error, info?.componentStack);
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    return (
      <div className={`panel crash ${this.props.className || ""}`}>
        <div className="crash-body">
          <AlertTriangle size={22} className="t-bad" />
          <div className="crash-title">{this.props.name || "This panel"} hit an error</div>
          <code className="crash-msg">{String(error?.message || error)}</code>
          <button className="btn btn-ghost btn-sm" onClick={() => this.setState({ error: null })}>
            <RotateCcw size={14} /> Reset panel
          </button>
        </div>
      </div>
    );
  }
}
