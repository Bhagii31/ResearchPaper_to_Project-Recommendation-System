export const metadata = {
  title: "Paper-to-Project Recommendation ",
  description: "Turn research papers into buildable ML projects, datasets, and learning resources.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{ margin: 0 }}>{children}</body>
    </html>
  );
}