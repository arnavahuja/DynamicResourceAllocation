export default function Button({ variant = "primary", children, ...props }) {
  const cls = "btn" + (variant === "ghost" ? " ghost" : variant === "danger" ? " danger" : "");
  return (
    <button className={cls} {...props}>
      {children}
    </button>
  );
}
