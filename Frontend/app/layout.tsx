export const metadata = {
  title: "Paper2Project",
  description: "Upload a paper PDF and get project ideas, datasets, and YouTube resources."
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
